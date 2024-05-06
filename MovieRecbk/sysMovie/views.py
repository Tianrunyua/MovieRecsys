from django.shortcuts import render
from django.http import JsonResponse
import json
import jwt
import time
from .models import User_table, Historyzhanshi_table, Movies_table, History_table  # 定义在model.py的Django模型，它们分别映射到数据库中的一个表。
import numpy as np
from sysMovie.IPTVModel import MultiGraph
import torch
import torch.nn.functional as F


# EGCN 用于计算文本/图像相似度
# 点击电影海报--进入电影详情页--页面下方展示与请求的电影ID（文本、图像）相似度高的电影列表
class Net:
    def __init__(self):
        # 本地
        self.device = torch.device("cpu")
        # 服务器
        # self.device = torch.device("cuda")  # 强制使用CUDA

        seed = 1                          # 设置随机数种子
        np.random.seed(seed)              # 设置NumPy的随机数种子
        self.seed = seed

        # 设置数据信息
        self.data_path = './sysMovie/MFMGNN_IPTVRS/Data/'
        self.dataset = 'IPTV'
        self.num_user = 143916
        self.num_item = 6259

        # 模型和训练参数
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.construction = 'weighted_sum'
        self.dim_latent = 128
        self.batch_size = 1024
        self.aggr_mode = 'add'           # 聚合模式
        self.num_CFGCN_layer = 1
        self.num_itemgraph_layer = 1
        self.has_id = True
        self.topk_user = 10
        self.pre_num = 2


        # 加载数据集和特征
        ## u-i
        self.user_item_dict_train = np.load(self.data_path + self.dataset + '/smalldata/user_item_dict_train.npy',allow_pickle=True).item()
        ## train val test
        self.edge_index = np.load(self.data_path + self.dataset + '/smalldata/train.npy', allow_pickle=True).T
        self.val_dataset = np.load(self.data_path + self.dataset + '/smalldata/val_full.npy', allow_pickle=True)
        self.test_dataset = np.load(self.data_path + self.dataset + '/smalldata/test_full.npy', allow_pickle=True)
        ## feat_v t   文本和图像的预训练特征向量
        self.v_feat = np.load(self.data_path + self.dataset + '/feat_v.npy', allow_pickle=True)
        self.t_feat = np.load(self.data_path + self.dataset + '/feat_t.npy', allow_pickle=True)
        ## graph_user  用户二部图
        self.user_graph = np.load(self.data_path + self.dataset +"/user_graph_sample.npy").tolist()
        ## graph_edge_index_com v t  # 加载项目图的边索引
        self.item_graph_edge_index_com = np.load(self.data_path + self.dataset +"/edge_com.npy")
        self.item_graph_edge_index_v = np.load(self.data_path + self.dataset +"/edge_sim_v.npy")
        self.item_graph_edge_index_t = np.load(self.data_path + self.dataset +"/edge_sim_t.npy")
        ## sim v t com
        self.simv_itemdict = np.load(self.data_path + self.dataset +"/simv_itemdict.npy",allow_pickle=True).item()
        self.simt_itemdict = np.load(self.data_path + self.dataset +"/simt_itemdict.npy",allow_pickle=True).item()
        self.itemcom_dict = np.load(self.data_path + self.dataset +"/item_graph_dict.npy",allow_pickle=True).item()
        # 特征初始化
        self.features = [self.v_feat, self.t_feat]

        # 模型初始化，加载模型权重
        self.model = MultiGraph(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item,
                                self.aggr_mode, self.construction, self.num_CFGCN_layer, self.num_itemgraph_layer,
                                self.has_id,
                                self.dim_latent, self.weight_decay, self.user_item_dict_train, self.dataset,
                                self.item_graph_edge_index_com, self.item_graph_edge_index_v,
                                self.item_graph_edge_index_t,
                                self.topk_user, self.pre_num, self.user_graph, device=self.device).to(self.device)

        # 本地
        model_state_dict = torch.load('./sysMovie/MFMGNN_IPTVRS/savemodel/model.pth', map_location ='cpu')
        # 服务器
        # model_state_dict = torch.load('./sysMovie/MFMGNN_IPTVRS/savemodel/model.pth')
        self.model.load_state_dict(model_state_dict)
        # print('module weights loaded....')

    def update_graph(self, curuser, curuser_history_list):
        '''更新历史记录更新后的二部图'''
        # 将当前用户的浏览历史添加到边索引列表中
        edge_index_list = self.edge_index.tolist()
        new_edge_index = np.array(edge_index_list+curuser_history_list)
        return new_edge_index

    def get_itemfeat(self):
        # 计算项目特征
        self.v_feat_hero, self.t_feat_hero = self.model.computer_itemfeature()

    def preget_modalatt(self):
        # 预先计算模态注意力
        self.u_pre_nousergraph, self.v_embd_att, self.t_embd_att = self.model.computer_itematt(self.edge_index,self.v_feat_hero,self.t_feat_hero)
        return self.v_embd_att[:self.num_user], self.t_embd_att[:self.num_user]

    def computer_modalatt(self, curuser, curitem):
        # 计算当前用户和项目的模态注意力
        index = torch.tensor(self.user_graph)[curuser].tolist()
        u_features = self.u_pre_nousergraph[index]
        user_matrix = self.model.user_weight_matrix[curuser]
        u_pre = torch.matmul(user_matrix, u_features)
        urepfinal = self.u_pre_nousergraph[curuser] + u_pre.squeeze()
        t_att = torch.matmul(urepfinal, self.t_embd_att[curitem])
        v_att = torch.matmul(urepfinal, self.v_embd_att[curitem])
        att = F.softmax(torch.cat((v_att, t_att), dim=0), dim=0).tolist()
        return att[0], att[1]

    def run(self, cur_user, curuser_history_list):
        # 更新二部图，其他图也可以在这里更新
        update_edge_index = self.update_graph(cur_user, curuser_history_list)
        # 获取推荐结果
        ranklist = self.model.forward(cur_user, update_edge_index, self.user_graph,self.v_feat_hero,self.t_feat_hero)
        return ranklist

egcn = Net()
egcn.get_itemfeat()
egcn.preget_modalatt()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


# Login.vue的方法，得到当前时间
def getLocalTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Login.vue的方法，登陆验证
def login_func(request):   # 参数为用户的输入：JSON.stringify(this.loginForm)
    print("here")
    userinput_json_str = request.body
    userinput_json_dict = json.loads(userinput_json_str)

    # 获取到用户输入的用户名和密码
    userinput_userName = userinput_json_dict.get("userName", None)
    userinput_password = userinput_json_dict.get("password", None)

    if userinput_userName and userinput_password:
        # 判断用户名是否一致，
        try:
            get_user_from_table = User_table.objects.get(username=userinput_userName)
        except:
            return JsonResponse({'code': 500, 'msg': '用户名不存在/用户名错误'})

        # 若存在，则判断密码是否一致
        if get_user_from_table.password == userinput_password:
            # 用户名和密码都一致，保存token到表中
            b = {'username': userinput_userName, 'password': userinput_password}
            token = jwt.encode(payload=b, key='sercet', algorithm='HS256')
            get_user_from_table.token = token
            get_user_from_table.save()

            # 获得系统登入情况，写入到展示表中
            user_who_login = User_table.objects.filter(username=userinput_userName)[0]
            Historyzhanshi_table(name=user_who_login.username, uid=int(user_who_login.id), time=getLocalTime(), action="登录", item_name="", bastags="").save()

            return JsonResponse({'code': 200, 'data':{'msg': '操作成功', 'token': token, 'sysUser': userinput_userName}})
        else:
            return JsonResponse({'code': 500, 'msg': '密码错误'})
    else:
        return JsonResponse({'code': 500, 'msg': '用户名或密码错误'})


# Welcome.vue的方法，登出
def quit_func(request):
    username = request.GET.get("username")
    # 记录登出信息
    userone = User_table.objects.filter(username=username)[0]
    Historyzhanshi_table(name=userone.username, uid=int(userone.id), time=getLocalTime(), action="退出", item_name="",
                   bastags="").save()
    # 保存code200信息，返回
    result = dict()
    result['data'] = dict()
    result['data']['code'] = 200
    return JsonResponse(result)


# Home.vue的方法，返回榜单
def rankingList_func(request):   # 传入榜单查询信息
    # 接口传入的page参数
    _page_id = int(request.GET.get("pageNum"))
    _page_size = int(request.GET.get("pageSize"))

    result = dict()
    result["code"] = 1
    result["data"] = dict()

    _list = list()
    MovieInfo = Movies_table.objects.order_by("-Watchtimes")    # __contains模糊查询
    total = MovieInfo.__len__()

    for one in MovieInfo[(_page_id - 1) * _page_size: _page_id * _page_size]:
        _list.append(
            {
                "movieName": one.name,
                "movie_id": one.id,
                "tag": one.bastags,
                "releaseDate":one.releaseyear,
                "watchtimes":one.Watchtimes,
                "area":one.originalcountry
            }
        )
    result["data"]= _list
    result["total"] = total
    return JsonResponse(result)


# Home.vue的方法，返回电影列表
# 得到当前类别的排名靠前的电影进行首页展示
def HomeList_func(request):    # 传入电影查询信息
    # 接口传入的page参数
    _page_id = int(request.GET.get("pageNum"))
    _page_size = int(request.GET.get("pageSize"))
    _MovieClass = request.GET.get("MovieClass")

    result = dict()
    result["code"] = 200
    result["data"] = dict()

    _list = list()
    # print(_MovieClass)
    # print(_page_id)
    if _MovieClass == '经典':
        MovieInfo = Movies_table.objects.order_by("-Watchtimes")
    else:
        MovieInfo = Movies_table.objects.filter(bastags__regex=r"."+_MovieClass+".")|Movies_table.objects.filter(bastags__startswith=_MovieClass).order_by("-Watchtimes") #__contains模糊查询
    # MovieIds = Movies.objects.filter(name__contains=_MovieClass).order_by("-Watchtimes") #__contains模糊查询
    total = MovieInfo.__len__()
    for one in MovieInfo[(_page_id - 1) * _page_size: _page_id * _page_size]:
        _list.append(
            {
                "movieName": one.name,
                "movie_id": one.id,
                "tag": one.bastags,
                "releaseDate": one.releaseyear,
                "watchtimes": one.Watchtimes,
                "area": one.originalcountry,
                "duration": one.duration,
                "movieIntroduction": one.description,
                "movieBoxOffice": one.Watchtimes * 30
            }
        )
    result["data"] = _list
    result["total"] = total
    return JsonResponse(result)


# movie页面的方法，获取所有电影类别标签
def getAllTag_func(request):
    """
    从数据库中检索电影标签，统计每个标签的出现次数，按出现次数降序排序，
    然后创建一个包含最多29个最常见标签的列表，并将其作为JSON响应返回。
    """
    alltags = Movies_table.objects.values("bastags")
    categoryList = []
    categoryDict = {}
    result = dict()
    print(alltags)
    for tagstr in alltags:
        tags = tagstr['bastags'].split('|')
        for tag in tags:
            if tag in categoryDict.keys():
                categoryDict[tag] += 1
            else:
                categoryDict[tag] = 1
    # 对tag根据出现次数排序
    categoryDict_sort = dict(sorted(categoryDict.items(), key=lambda e: -e[1]))
    for index, tag in enumerate(categoryDict_sort.keys()):
        categoryList.append({
            "movieCategoryId": index+1,
            "movieCategoryName": tag
        })
    result["data"] = categoryList[:29]
    # result["data"] = categoryList[:10]
    result["code"] = 200
    return JsonResponse(result)


# movie页面的方法，获取电影列表
def findtag_func(request):
    """
    接收HTTP请求并根据请求参数（如页码、每页大小、电影分类、排序字段和升序/降序标志）
    查询和排序电影数据库记录，然后分页返回电影信息的JSON响应。
    """
    _page_id = int(request.GET.get("pageNum"))
    _page_size = int(request.GET.get("pageSize"))
    _MovieClass = request.GET.get("movieCategoryId")
    _orderByColumn = request.GET.get("orderByColumn")
    _isAsc = request.GET.get("isAsc")

    if _isAsc == "desc":   # 指示升序或降序
        _isAsc = "-"
    else:
        _isAsc = ""

    result = dict()
    result["code"] = 200
    result["data"] = dict()
    _list = list()

    if _MovieClass == '全部':   # 则查询所有电影记录，并根据 _orderByColumn 和 _isAsc 进行排序。
        MovieInfo = Movies_table.objects.order_by(_isAsc+_orderByColumn)
    else:    # 则使用正则表达式和 startswith 方法查询特定分类的电影记录，并进行排序。
        # 包含 _MovieClass 的、以 _MovieClass 指定的分类标签开始的、并根据 orderByColumn 和 isAsc 参数指定的字段和方向进行排序。
        MovieInfo = Movies_table.objects.filter(bastags__regex=r"."+_MovieClass+".") | Movies_table.objects.filter(bastags__startswith=_MovieClass).order_by(_isAsc+_orderByColumn) #__contains模糊查询

    total = MovieInfo.__len__()
    for one in MovieInfo[(_page_id - 1) * _page_size: _page_id * _page_size]:
        _list.append(
            {
                "movieName": one.name,
                "movie_id": one.id,
                "tag": one.bastags,
                "releaseDate":one.releaseyear,
                "watchtimes":one.Watchtimes,
                "area":one.originalcountry,
                "duration":one.duration,
                "movieIntroduction":one.description
            }
        )
    result["data"]= _list
    result["total"] = total
    return JsonResponse(result)   # 以分页的形式返回给前端


# movieinfo页面的方法，获取电影信息
# 根据类别查询单部电影信息
def getMovieInfo_func(request):
    # 接口传入的page参数
    _Movie_id = int(request.GET.get("movie_id"))
    _Username = request.GET.get("username")
    # print(_Username)
    # print(_Movie_id)
    print("此处还未跳转吧")

    result = dict()
    result["code"] = 1
    result["data"] = dict()
    _list = list()

    one = Movies_table.objects.filter(id=_Movie_id)[0]  # __contains模糊查询
    if one.Watchtimes == None: one.Watchtimes = 1
    result["data"] = {
                "movieName": one.name,
                "movie_id": one.id,
                "tag": one.bastags,
                "releaseDate":one.releaseyear,
                "watchtimes":one.Watchtimes,
                "area":one.originalcountry,
                "duration":one.duration,
                "movieIntroduction":one.description,
                "movieBoxOffice":one.Watchtimes * 30
            }
    return JsonResponse(result)


# movieinfo页面的方法，获取观看记录后，写入记录
def write_to_mysql(name="",uid="",mapitemid="", time="", action="", item_name="",iptv_itemid="", bastags=""):
    # 看电影记录同时写入到两张历史表中
    History_table(name=name,uid=uid,mapitemid=mapitemid, time=time, action=action, item_name=item_name,iptv_itemid=iptv_itemid, bastags=bastags).save()
    Historyzhanshi_table(name=name,uid=uid, time=time,item_name=item_name, action=action,bastags=bastags).save()
    # print("写入成功")


# movieinfo页面的方法，获取观看记录
def watchrecord_func(request):
    # 接口传入的page参数
    _Movie_id = int(request.GET.get("movie_id"))
    _Username = request.GET.get("username")
    # print(_Username)
    # print(_Movie_id)
    result = dict()
    result["code"] = 1
    result["data"] = dict()
    _list = list()
    one = Movies_table.objects.filter(id=_Movie_id)[0] # __contains模糊查询
    # print(one)
    write_to_mysql(
        name=_Username,
        uid =_Username[4:],
        time=getLocalTime(),
        action="查看",
        item_name=one.name,
        mapitemid=one.mapitemid,
        iptv_itemid=one.id,
        bastags=one.bastags
    )
    result["data"] = {
                "movieName": one.name,
                "movie_id": one.id,
                "tag": one.bastags,
                "releaseDate":one.releaseyear,
                "watchtimes":one.Watchtimes,
                "area":one.originalcountry,
                "duration":one.duration,
                "movieIntroduction":one.description,
                "movieBoxOffice":one.Watchtimes* 30
            }
    return JsonResponse(result)


# movieinfo页面的方法，获取文本相似度高的电影列表
def getTsimMovies_func(request):
    # 从GET请求参数中获取电影ID，并将其转换为整数
    _Movie_id = int(request.GET.get("movie_id"))

    result = dict()
    result["code"] = 1       # 设置返回码为1，表示成功
    result["data"] = dict()  # 初始化数据字段，稍后将存放电影列表
    _list = list()           # 初始化一个空列表_list，用于存放电影信息字典

    # 获取文本相似度高电影(0-6259)
    # 获取与请求的电影ID相似度高的电影列表（0-6259）
    # 注意列表的索引是从0开始的，所以需要对电影ID进行-1操作
    simMovieid_list = egcn.simt_itemdict[_Movie_id-1][0][:16]  # 获取相似电影的ID列表，取前16个
    simMovieval_list = egcn.simt_itemdict[_Movie_id-1][1][:16]  # 获取相似电影的相似度值列表，取前16个

    # 使用zip函数同时遍历电影ID列表和相似度值列表
    for movieid, simval in zip(simMovieid_list, simMovieval_list):
        # 通过电影ID从数据库中查询电影对象
        one = Movies_table.objects.filter(id=movieid+1)[0]  # 注意+1
        # 构建包含电影信息的字典，并添加到_list列表中
        _list.append( {
            "movieName": one.name,
            "movie_id": one.id,
            "tag": one.bastags,
            "releaseDate": one.releaseyear,
            "area": one.originalcountry,
            "simval":round(simval*100, 2),
            "iswatched": False
        })
    # 将_list列表赋值给result字典中的data字段
    result["data"] = _list
    # 返回一个JsonResponse对象，将result字典转换为JSON格式的响应体
    return JsonResponse(result)


# movieinfo页面的方法，获取图片相似度高的电影列表 思路和上一个方法一模一样
def getVsimMovies_func(request):
    _Movie_id = int(request.GET.get("movie_id"))

    result = dict()
    result["code"] = 1
    result["data"] = dict()
    _list = list()

    # 获取相似度高电影(0-6259)
    simMovieid_list = egcn.simv_itemdict[_Movie_id-1][0][:16]
    simMovieval_list = egcn.simv_itemdict[_Movie_id-1][1][:16]

    for movieid, simval in zip(simMovieid_list, simMovieval_list):
        one = Movies_table.objects.filter(id=movieid+1)[0]  # __contains模糊查询
        _list.append( {
            "movieName": one.name,
            "movie_id": one.id,
            "tag": one.bastags,
            "releaseDate": one.releaseyear,
            "area": one.originalcountry,
            "simval":round(simval*100, 2),
            "iswatched": False

        })
    result["data"] = _list
    return JsonResponse(result)


# movieinfo页面的方法，获取高共现度相关电影列表的方法
def getcomMovies_func(request):
    _Movie_id = int(request.GET.get("movie_id"))

    result = dict()
    result["code"] = 1
    result["data"] = dict()
    _list = list()

    # 获取高共现度的电影(0-6259)
    simMovieid_list = egcn.itemcom_dict[_Movie_id - 1][0][:16]  # 注意-1
    simMovieval_list = egcn.itemcom_dict[_Movie_id - 1][1][:16]  # 注意-1

    #获取用户观看历史，判断是否观看过该部影片
    for movieid, comval in zip(simMovieid_list, simMovieval_list):
        one = Movies_table.objects.filter(id=movieid + 1)[0]  # 注意+1
        _list.append({
            "movieName": one.name,
            "movie_id": one.id,
            "tag": one.bastags,
            "releaseDate": one.releaseyear,
            "area": one.originalcountry,
            "comval": comval,
            "iswatched": False
        })
    result["data"] = _list
    return JsonResponse(result)
