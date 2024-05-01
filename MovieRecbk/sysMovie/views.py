from django.shortcuts import render
from django.http import JsonResponse
import json
import jwt
import time
from .models import User_table, Historyzhanshi_table  # 定义在model.py的Django模型，它们分别映射到数据库中的一个表。
import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)


# 得到当前时间
def getLocalTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# 登陆验证
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
