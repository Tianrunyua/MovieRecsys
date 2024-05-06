<template>
  <div>
    <!-- 轮播图 -->
    <el-carousel :height="carouselHeight">
      <!-- 循环渲染每个轮播图项，相对url -->
      <el-carousel-item v-for="item in posterList" :key="item.url">
        <img :src="item.url" alt/>
      </el-carousel-item>
     </el-carousel>

    <!-- 整个页面的容器 -->
    <div class="whole">

      <!-- 左侧内容容器 -->
      <div class="left">

        <!-- 电影分类面板1 -->
        <div class="panel">
          <!-- 面板头部1，分类标题和查看全部 -->
          <div class="panel-header">
            <h2 style="color: #ef4238">{{this.tag1}}</h2>
            <!-- 查看全部链接，使用Element UI的el-link组件，点击时触发totag1方法 -->
            <el-link :underline="false" @click="totag1">全部<i class="el-icon-arrow-right el-icon--right"></i></el-link>
          </div>

          <!-- 面板内容1，循环渲染电影列表项 -->
          <div class="panel-content">
            <movie-item :movieItem="item" v-for="(item, index) in ClassicMovieList" :key="index"></movie-item>
          </div>
        </div>

        <!-- 电影分类面板2 -->
        <div class="panel">
          <div class="panel-header">
            <h2 style="color: #f8a605">{{this.tag2}}</h2>
            <el-link :underline="false" @click="totag2">全部<i class="el-icon-arrow-right el-icon--right"></i></el-link>
             </div>
            <div class="panel-content">
              <movie-item :movieItem="item" v-for="(item, index) in ActionMovieList" :key="index"></movie-item>
            </div>
        </div>

        <!-- 电影分类面板3 -->
        <div class="panel">
          <div class="panel-header">
            <h2 style="color: #e7ea8b">{{this.tag3}}</h2>
            <el-link :underline="false" @click="totag3">全部<i class="el-icon-arrow-right el-icon--right"></i></el-link>
             </div>
            <div class="panel-content">
              <movie-item :movieItem="item" v-for="(item, index) in ComedyMovieList" :key="index"></movie-item>
            </div>
        </div>

        <!-- 电影分类面板4 -->
        <div class="panel">
          <div class="panel-header">
            <h2 style="color: #b5e89f">{{this.tag4}}</h2>
            <el-link :underline="false" @click="totag4">全部<i class="el-icon-arrow-right el-icon--right"></i></el-link>
             </div>
            <div class="panel-content">
              <movie-item :movieItem="item" v-for="(item, index) in SportsMovieList" :key="index"></movie-item>
            </div>
        </div>
      </div>

      <!-- 右侧内容容器 -->
      <div class="right">
        <!-- 榜单面板，包括头部的榜单标题、内容的循环渲染榜单项 -->
        <div class="panel">

          <div class="panel-header">
            <h2 style="color: #ffb400">榜单</h2>
          </div>

          <div class="panel-content">
            <div class="board" v-for="(item, index) in totalBoxOfficeList">

              <!-- 榜单左侧，显示排名 -->
              <div class="board-left">
                <i class="board-index">{{index+1}}</i>
              </div>

              <!-- 榜单中间，显示电影名称和电影详细信息 -->
              <div class="board-middle">
                <a @click="gotomovie(item.movie_id)">
                <!-- a :href="'/movieInfo/' + item.movie_id" -->
                  <p class="name">{{ item.movieName }}</p>
                </a>
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>



<script>
import movieItem from './func_module/movie-item'
import moment from 'moment'  // 时间处理库

export default {
  name: "Home",
  components: {
    movieItem   // 注册 movieItem 组件，使其可以在模板中使用
  },

  data() {
    return {
      // 从 sessionStorage 获取标签信息
      tag1 : window.sessionStorage.getItem('tag1'),
      tag2 : window.sessionStorage.getItem('tag2'),
      tag3 : window.sessionStorage.getItem('tag3'),
      tag4 : window.sessionStorage.getItem('tag4'),

      // 为不同电影类型定义查询信息，包括分页和电影类别
      queryInfo1: {
        total: 0,
        pageSize: 4,
        pageNum: 1,
        MovieClass: window.sessionStorage.getItem('tag1')
      },
      queryInfo2: {
        total: 0,
        pageSize: 4,
        pageNum: 1,
        MovieClass:  window.sessionStorage.getItem('tag2')
      },
      queryInfo3: {
        total: 0,
        pageSize: 4,
        pageNum: 1,
        MovieClass:  window.sessionStorage.getItem('tag3')
      },
      queryInfo4: {
        total: 0,
        pageSize: 4,
        pageNum: 1,
        MovieClass:  window.sessionStorage.getItem('tag4')
      },

      // 定义榜单查询信息
      queryInforank:{
        pageSize: 20,
        pageNum: 1
      },

      // 定义轮播图的图片列表
      posterList: [
        {url: require('../assets/pic/carousel1.jpg')},
        {url: require('../assets/pic/carousel2.jpg')},
        {url: require('../assets/pic/carousel3.jpg')},
        {url: require('../assets/pic/carousel4.jpg')}
      ],

      // 定义不同电影类型列表的数组
      ClassicMovieList: [],
      ActionMovieList: [],
      ComedyMovieList: [],
      SportsMovieList: [],

      // 定义轮播图高度的变量
      carouselHeight: '',

      // 定义总票房榜单的数组
      totalBoxOfficeList: [],
    }
  },
  created() {
    // Vue 实例的生命周期钩子，在实例创建后被调用
    // 获得轮播图高度
    this.getHeight()
    // 异步方法获取经典电影列表
    this.getclassicMovieList()
    this.getActionMovieList()
    this.getComedyMovieList()
    this.getSportsMovieList()
    // 榜单
    this.getTotalBoxOfficeList()
  },
  methods:{
    // 标签召回获取电影列表的方法
    async getclassicMovieList() {
      const { data : res } = await this.$axios.get('sysMovie/HomeList', {params: this.queryInfo1})
      this.ClassicMovieList = res.data   // 更新经典电影列表
      this.total = res.total
    },
    async getActionMovieList() {
      const { data : res } = await this.$axios.get('sysMovie/HomeList', {params: this.queryInfo2})
      this.ActionMovieList = res.data
      this.total = res.total
    },
    async getComedyMovieList() {
      const { data : res } = await this.$axios.get('sysMovie/HomeList', {params: this.queryInfo3})
      this.ComedyMovieList = res.data
      this.total = res.total
    },
    async getSportsMovieList() {
      const { data : res } = await this.$axios.get('sysMovie/HomeList', {params: this.queryInfo4})
      this.SportsMovieList = res.data
      this.total = res.total
    },

    // 方法计算轮播图的高度
    getHeight() {
      let clientWidth =   `${document.documentElement.clientWidth}`
      clientWidth *= 0.8
      this.carouselHeight = clientWidth / 1700 * 520 + 'px'  // 更新轮播图高度
    },

    // 异步方法获取总票房榜单
    async getTotalBoxOfficeList(){
      const {data: resp} = await this.$axios.get('sysMovie/rankingList', {params: this.queryInforank})  // 传入榜单查询信息
      console.log(resp)
      if(resp.code !== 1) return this.$message.error(resp.msg)
      console.log(resp.data)
      this.totalBoxOfficeList = resp.data
    },

    // 点击“全部”，触发下方方法处理点击标签1的逻辑
    totag1() {
      window.sessionStorage.setItem('default_tag', this.tag1)
      window.sessionStorage.setItem('homestep', 1) //用于验证从主页全部跳到的tag页
      console.log(window.sessionStorage.getItem('homestep'))
      this.$router.push('/movie')
    },
    totag2() {
      window.sessionStorage.setItem('default_tag', this.tag2)
      window.sessionStorage.setItem('homestep', 1)
      this.$router.push('/movie')
    },
    totag3() {
      window.sessionStorage.setItem('default_tag', this.tag3)
      window.sessionStorage.setItem('homestep', 1)
      this.$router.push('/movie')
    },
    totag4() {
      window.sessionStorage.setItem('default_tag', this.tag4)
      window.sessionStorage.setItem('homestep', 1)
      this.$router.push('/movie')
    },
    gotomovie :function (opt){
      this.$router.push('/movieInfo/' + opt)
    }
  }
}
</script>


<style scoped>

.el-carousel {
  width: 80%;
  margin: 30px auto;
}

.el-carousel__item > img {
  width: 100%;
  height: auto;
}

.whole{
  width: 1200px;
  margin: 30px auto;
  display: flex;
}

.left{
  width: 80%;
}

.right{
  width: 20%;
  margin-left: 100px;
}

h2{
  font-size: 26px;
}

.panel-header > a{
  text-align: center;
  text-decoration: none;
  color: #999;
  padding-right: 14px;
  /*background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAOCAYAAAASVl2WAAAABmJLR0QA/wD/AP+gvaeTAAAAv0lEQVQY013RTUpDQRAE4G8eghcR8ScgKCIugpJFjuIjqAvBc7jxj0muEnCjiIQQJOImB3GnbnpkfL1qpqqrunpSzvkDPxjhGdq2VarBF3q4wRHknP8RzvCEQzzguCalaHZwiwHecY6XogCf8TjFHh7Rh9Tx3AylIZa4TgWpSBuY4BSrYlFXKsr4bjrTW5HkJJa9SBW4jbtukmKxG5MDLOKqfzEPcB9LzQN8LSdfwxj7eMMlZvV/NFiPzFddEH4Bt5Y1mf3fnDwAAAAASUVORK5CYII=) no-repeat 100%*/
}


.panel-header{
  display: flex;
  justify-content: space-between;
  align-items: center;
  /*margin-right: 20px;*/
  /*margin-left: 20px;*/
}

.movie-item{
  margin-left: 0;
  margin-right: 30px;
}

.movie-item:nth-child(4n){
  margin-right: 0;
}

.board{
  display: flex;
  margin: 10px 10px;
}

.board-left{
  display: flex;
  align-items: center;
}

.board-middle{
  display: flex;
  /*align-items: center;*/
  /*justify-content: center;*/
  margin-left: 10px;
  width: 150px;
  font-size: 18px;
}

.board-middle > a{
  text-decoration: none;
  color: #333;
}

.board-right{
  display: flex;
  font-size: 14px;
  font-weight: 700;
  color: #ffb400;
  margin-left: 40px;
  align-items: center;
}

.board-index{
  color: #ffb400;
  display: inline-block;
  width: 50px;
  height: 50px;
  line-height: 50px;
  text-align: center;
  font-size: 18px;
  /*font-weight: 700;*/
  align-items: center;
}

.panel-content{
  margin: 0px 0px 50px 0px;
}
.name, .star, .releaseTime{
  margin-top: 8px;
  margin-bottom: 8px;
}
</style>

