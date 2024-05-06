<template>
  <el-container>
    <el-main>
      <!-- 类别查询 -->
      <div class="tags-container">
        <div class="tags-line">

          <div class="tags-title">类型:</div>

          <!-- 单选按钮组，用于选择电影类别 -->
          <el-radio-group v-model="categoryRadio">
            <!-- 默认选项，选中时表示“全部”类型 -->
            <el-radio-button :label='movieCategoryNameinit' key='movieCategoryNameinit'>全部</el-radio-button>
            <!-- 循环渲染每个类别的单选按钮 -->
            <el-radio-button :label="item.movieCategoryName" v-for="item in categoryList" :key="item.movieCategoryName">{{item.movieCategoryName}}</el-radio-button>
          </el-radio-group>

        </div>
      </div>

      <!-- 按条件排序 -->
      <div class="order-by-container">
        <!-- 按观看次数排序的单选按钮 -->
        <el-radio v-model="orderByColumn" label="Watchtimes">按观看次数排序</el-radio>
        <!-- 按时间排序的单选按钮 -->
        <el-radio v-model="orderByColumn" label="releaseyear">按时间排序</el-radio>
      </div>

      <div>
        <div class="movie-list">
          <!-- 循环渲染每个电影项 -->
          <movie-item :movieItem="item" v-for="item in movieList" :key="item.movie_id"></movie-item>
        </div>

        <div class="no-cinema" v-if="sorry">
          <p>抱歉，没有找到相关结果，请尝试用其他条件筛选。</p>
        </div>

        <!-- 分页助手容器 -->
        <div class="pageHelper">
          <!-- 分页组件，用于在不同页面间导航 -->
          <el-pagination
              @current-change="handleCurrentChange"
              :current-page="pageNum"
              :page-size="pageSize"
              background
              layout="prev, pager, next"
              :total="total">
          </el-pagination>
        </div>
      </div>
    </el-main>
  </el-container>
</template>





<script>
import movieItem from './func_module/movie-item'

export default {
  name: "Movie",
  components:{
    movieItem
  },
  data(){
    return {
      categoryRadio: '',
      movieCategoryNameinit:'全部',
      categoryList: [],
      orderByColumn: 'Watchtimes',
      total: 0,
      pageSize: 30,
      pageNum: 1,
      movieList: [],
      sorry: false,
      queryInfo:{
        curtag: '全部'
      }
    }
  },

  // 计算属性，用于返回筛选和排序条件
  computed: {
    listenChange () {
      const {categoryRadio, orderByColumn} = this
      return {categoryRadio, orderByColumn}
    }
  },

  created() {
    this.initcatetag()       // 调用初始化当前选中类别标签的方法。
    this.getCategoryList()   // 获取所有可用的电影类别标签列表。
    this.getMovieList()      // 获取当前页的电影列表。
  },

  methods: {
    async getCategoryList() {
      // 异步方法，按出现次数降序排序 获取所有电影类别标签。
      const { data : res } = await this.$axios.get('sysMovie/getAllTag', {params: this.queryInfo})  // 发送 GET 请求到服务器，获取所有电影类别标签。
      console.log(res)
      if(res.code !== 200) return this.$message.error('获取服务器信息失败')
      this.categoryList = res.data
      console.log(this.categoryList)
    },

    async getMovieList(){   // 异步方法，用于根据当前的筛选和排序条件获取电影列表。
      let queryInfo_tag = {   // 构造请求服务器获取电影列表的参数对象。
        movieCategoryId: this.categoryRadio,
        orderByColumn: this.orderByColumn,
        pageSize: this.pageSize,
        pageNum: this.pageNum,
        isAsc: 'desc'
      }
      console.log(queryInfo_tag)
      // 发送 GET 请求到服务器，获取电影列表。
      const { data : res } = await this.$axios.get('sysMovie/findtag', {params: queryInfo_tag})
      this.movieList = res.data       // 将获取到的电影列表数据赋值给 movieList。
      this.total = res.total          // 将获取到的总项数赋值给 total。
      if(this.movieList.length === 0) this.sorry = true   // 如果电影列表为空，则显示提示信息。
      else this.sorry = false
    },

    handleCurrentChange(newPage) {   // 事件处理函数，用于处理分页组件的页码改变事件。
      this.pageNum = newPage
      this.getMovieList()     // 重新获取当前页的电影列表。
    },

    initcatetag(){
      // 方法，用于初始化当前选中的类别标签。
      let curlocaltag = window.sessionStorage.getItem('default_tag')  // 从 sessionStorage 获取之前选中的默认标签。
      console.log(curlocaltag)
      console.log('homestep')
      console.log(window.sessionStorage.getItem('homestep'))
      if(window.sessionStorage.getItem('homestep')==0)  // 如果 sessionStorage 中的 homestep 为 0，则默认选中“全部”。
      {
        this.categoryRadio = '全部'
        console.log('1')
      }
      else   // 否则，使用 sessionStorage 中存储的标签。
      {
        this.categoryRadio = curlocaltag
        window.sessionStorage.setItem('homestep', 0)
        console.log('2')
      }

    }
  },

  watch:{
    '$route'(){    // 监听路由对象 $route 的变化。
      // 页面发生变化时初始化类别选项
      this.categoryRadio = 'categoryRadio'
      this.orderByColumn = 'releaseyear'
    },
    listenChange(){
      this.getMovieList()   // 当筛选或排序条件变化时，重新获取电影列表。
    }
  }
}
</script>



<style scoped>
.el-header{
  padding: 0px;
}

.nav-menu{
  display: flex;
  justify-content: space-around;
}

.el-menu-item{
  font-size: 16px;
}

.el-menu-item:hover{
  color: #FFFFFF !important;
}

.el-main{
  width: 1200px;
  margin: 0 auto;
}

.tags-container{
  border: solid #EEEEEE 1px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)
}

.tags-line{
  width: 100%;
  display: flex;
  padding: 10px;
  box-sizing: border-box;
  border-bottom: solid #EEEEEE 1px;
}

.tags-line:last-child{
  border-bottom: 0;
}

.tags-title {
  width: 5%;
  height: 24px;
  line-height: 24px;
  float: left;
  color: #999;
  font-size: 14px;
}

.el-radio-group{
  width: 92%;
  margin-left: 20px;
}

.el-radio-button {
  margin-right: 10px;
}

.el-radio-button:last-child{
  margin-right: 0;
}

.el-radio-button >>> .el-radio-button__inner {
  border: solid #eee 0px !important;
  border-radius: 50px !important;
}

.order-by-container{
  margin: 30px 10px 0 10px;
}
.pageHelper{
  display: flex;
  justify-content: center;
  margin: 40px 0px;
}
</style>
