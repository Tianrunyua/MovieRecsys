<template>
  <div class="movie-container">
    <div class="header">
      <div class="header-inner clearfix">

        <!-- 左侧电影信息，包含头像 -->
        <div class="movie-info-left">
          <!-- 电影头像图片 -->
          <div class="avatar-shadow">
            <img class="avatar" :src='"../assets/pic/poster/"+movieInfo.movie_id+".jpg"'>
          </div>
        </div>

        <!-- 右侧电影信息，包含电影名称、标签、地区、时长、上映日期和按钮 -->
        <div class="movie-info-right">
          <div class="movie-info-msg">
            <h1 class="movie-name">{{movieInfo.movieName}}</h1>
            <h1>&nbsp;</h1>

            <ul>
              <li>{{movieInfo.tag}}</li>
              <li>{{movieInfo.area}} / {{movieInfo.duration}}分钟</li>
              <li>{{movieInfo.releaseDate}} {{movieInfo.area}} 上映</li>
            </ul>
          </div>

          <!-- 电影操作按钮，如在线观看 -->
          <div class="movie-info-btn">
            <el-button class="buy-btn" type="primary" @click="watchrecord">在线观看</el-button>
          </div>

          <!-- 电影观看次数统计 -->
          <div class="movie-info-score">
            <div class="movie-index box-office-container">
              <span class="movie-index-title">观看次数</span>
              <div style="display: flex;align-items: flex-end;">
                <span class="box-office">{{movieInfo.watchtimes}}</span>
                <span class="unit">次</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 电影详细信息容器 -->
    <div class="movie-info-detail-container">
      <div class="movie-info-detail clearfix">
        <div class="main-content">
          <div class="crumbs-nav-container">
            <a href="/welcome">CUC电影</a> &gt;
            <a href="/movie" >电影</a> &gt;
            {{movieInfo.movieName}}
          </div>

          <!-- 标签页组件，包含电影介绍、相关项目等 -->
          <el-tabs v-model="activeName">
            <el-tab-pane label="介绍"  name="introduction">
              <div class="tab-body">
                <div class="module">
                  <div class="mod-title">
                    <h2>剧情简介</h2>
                  </div>
                  <div class="mod-content">
                    <span class="dra">
                      {{movieInfo.movieIntroduction}}
                    </span>
                  </div>
                </div>

                <!-- 图集 -->  <!-- 文本模态相关项目模块 -->
                <div class="module">
                  <div class="mod-title">
                    <h2>文本模态相关项目</h2>
                  </div>

                  <div class="mod-content">
                    <!-- 循环渲染相似电影项 -->
                    <div class="movie-list">
                    <simmovieItem :movieItem="item" v-for="item in tsimmovieList" :key="item.movie_id"></simmovieItem>
                    </div>

                    <!--画项目语义图或者展示相关项目以及相似度-->  <!-- 图片列表容器，预留用于展示图片或相关项目 -->
                    <div class="pictures-list"></div>
                  </div>

                </div>

                <!-- 图像模态相关项目模块 -->
                <div class="module">
                  <div class="mod-title">
                    <h2>图像模态相关项目</h2>
                  </div>
                  <div class="mod-content">
                    <!-- 循环渲染相似电影项 -->
                    <div class="movie-list">
                    <simmovieItem :movieItem="item" v-for="item in vsimmovieList" :key="item.movie_id"></simmovieItem>
                    </div>
                    <!-- 画项目语义图或者展示相关项目以及相似度-->
                    <div class="pictures-list"></div>
                  </div>
                </div>

                <!-- 高共现度相关项目模块 -->
                <div class="module">
                  <div class="mod-title">
                    <h2>高共现度相关项目</h2>
                  </div>
                  <div class="mod-content">
                    <div class="movie-list">
                    <commovieItem :movieItem="item" v-for="item in commovieList" :key="item.movie_id"></commovieItem>
                    </div>
                    <!--画项目语义图或者展示相关项目以及相似度-->
                    <div class="pictures-list">
                    </div>
                  </div>
                </div>
              </div>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </div>
  </div>
</template>



<script>
import simmovieItem from './func_module/simmovie-item'
import commovieItem from './func_module/commovie-item'
import moment from 'moment'

export default {
  name: "MovieInfo",
  components:{
    simmovieItem,
    commovieItem
  },

  data() {
    return {
      movieInfo: {
        // moviePictures: []
      },
      activeName: 'introduction',    // 定义当前活动的标签页名称
      colors: ['#99A9BF', '#F7BA2A', '#FF9900'],
      httpURL: this.global.base,     // 定义HTTP请求的基础URL
      vsimmovieList:[],       // 定义一个数组，用于存储视频相似电影列表
      tsimmovieList:[],      // 定义一个数组，用于存储文本相似电影列表
      commovieList:[]        // 定义一个数组，用于存储高共现度相关电影列表
    }
  },

  created() {
    this.getMovieInfo()
    this.getTsimItem()
    this.getVsimItem()
    this.getcomItem()   // 获取高共现度相关电影列表的方法
  },

  methods: {
    async getMovieInfo(){
      const _this = this
      let queryInfo1 = {
        movie_id: _this.$route.params.movie_id,   // 从路由参数中获取电影ID
        username: window.sessionStorage.getItem("loginUser")   // 从会话存储中获取用户名
      }
      const {data : res} = await this.$axios.get('sysMovie/getMovieInfo/',{params: queryInfo1})

      if(res.code !== 1) return this.$message.error('数据查询失败')
      this.movieInfo = res.data
      console.log(this.movieInfo.tag)
    },

    // 观看记录的方法
    async watchrecord(){
      const _this = this
      let queryInfo2 = {
        movie_id: _this.$route.params.movie_id,
        username: window.sessionStorage.getItem("loginUser")
      }
      const {data : res} = await this.$axios.get('sysMovie/watchrecord/',{params: queryInfo2})
      if(res.code !== 1){
        this.$alert('系统出错', '温馨提示', {
            confirmButtonText: '好的'})
      }
      else {
        this.$alert('成功观看', '温馨提示', {
            confirmButtonText: '好的'})
      }
    },

    //得到文本模态相关项目  获取文本相似电影列表的方法
    async getTsimItem(){
      const _this = this
      let queryInfo = {
        movie_id: _this.$route.params.movie_id,
        username: window.sessionStorage.getItem("loginUser")
      }
      const {data : res} = await this.$axios.get('sysMovie/getTsimMovies/',{params: queryInfo})
      // console.log(res.code)
      if(res.code !== 1) return this.$message.error('数据查询失败')
      this.tsimmovieList = res.data
    },

    // 获取图像相似电影列表的方法
    async getVsimItem(){
      const _this = this
      let queryInfo = {
        movie_id: _this.$route.params.movie_id,
        username: window.sessionStorage.getItem("loginUser")
      }
      const {data : res} = await this.$axios.get('sysMovie/getVsimMovies/',{params: queryInfo})
      // console.log(res.code)
      if(res.code !== 1) return this.$message.error('数据查询失败')
      this.vsimmovieList = res.data
    },

    // 获取高共现度相关电影列表的方法
    async getcomItem(){
      const _this = this
      let queryInfo = {
        movie_id: _this.$route.params.movie_id,
        username: window.sessionStorage.getItem("loginUser")
      }
      const {data : res} = await this.$axios.get('sysMovie/getcomMovies/',{params: queryInfo})
      console.log(res)
      if(res.code !== 1) return this.$message.error('数据查询失败')
      this.commovieList = res.data
    },
  }
}
</script>


<style scoped>
.movie-container{

}

a{
  text-decoration: none;
  cursor:pointer;
}

.header{
  padding: 0;
  /*width: 100%;*/
  width: auto;
  min-width: 1200px;
  background: url('../assets/pic/movie-info-background.jpg')
}

.header-inner{
  width: 1200px;
  margin: 0 auto;
  position: relative;
}

.clearfix::before, .clearfix::after{
  content: " ";
  display: table;
}

.clearfix::after{
  clear: both;
}

.movie-info-left{
  width: 300px;
  height: 370px;
  float: left;
  position: relative;
  top: 70px;
  overflow: hidden;
  z-index: 9;
}

.avatar-shadow{
  position: relative;
  margin: 0 30px;
  width: 240px;
  height: 330px;
  padding-bottom: 40px;
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPAAAAAyAgMAAAAfG76+AAAADFBMVEUAAAAAAAAAAAAAAAA16TeWAAAABHRSTlMOAgoGQ0SIeAAAADpJREFUSMdjGAWjYBRgAasoAAwdFACKbB7VPEI076YAUGbzfwrAqOYRormcAjCANodSAEY1j2oexJoBlx1+yE7RXIIAAAAASUVORK5CYII=) no-repeat bottom;
}

.avatar{
  border: 4px solid #fff;
  height: 322px;
  width: 232px;
}

.movie-info-msg{
  position: absolute;
  color: #fff;
  font-size: 14px;
  z-index: 1;
}

.movie-name{
  width: 900px;
  margin-top: 0;
  font-size: 26px;
  line-height: 32px;
  font-weight: 700;
  margin-bottom: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  max-height: 64px;
}

.movie-ename{
  width: 340px;
  font-size: 18px;
  line-height: 1.3;
  margin-bottom: 14px;
}

ul{
  width: 250px;
  list-style: none;
  padding-left: 0;
  margin-bottom: 20px;
}

ul li{
  margin: 12px 0;
  line-height: 100%;
}

.movie-info-btn{
  position: absolute;
  bottom: 20px;
}

.buy-btn{
  margin-top: 10px;
  width: 250px;
  height: 40px;
  font-size: 16px;
  text-align: center;
}

.movie-info-score{
  position: absolute;
  top: 145px;
  left: 342px;
}

.movie-index {
  margin-bottom: 16px;
  color: #fff;
}

.box-office-container{
  display: flex;
  flex-direction: column;
}

.movie-index-title{
  font-size: 12px;
  margin-bottom: 8px;
}

.movie-score-num{
  font-size: 30px;
  color: #ffc600;
  height: 30px;
  line-height: 30px;
  margin-right: 10px;
}

.score-container{
  display: flex;
  flex-direction: column;
  align-items: center;
  font-size: 12px;
}

.box-office{
  font-size: 30px;
}

.unit{
  font-size: 12px;
  padding-left: 1px;
  line-height: 24px;
}

.movie-info-right{
  width: auto;
  height: 300px;
  position: relative;
  margin-right: 30px;
  margin-left: 300px;
  margin-top: 70px;
}

.movie-info-detail-container{
  width:1300px;
  margin: 0 auto;
}

.movie-info-detail{
  margin-top: 80px;
}

.main-content {
  /*width: 730px;*/
  width: auto;
  float: left;
  margin-bottom: 20px;
}

.crumbs-nav-container {
  margin-bottom: 25px;
  color: #333;
  font-size: 25px;
}

.crumbs-nav-container a{
  color: inherit;
}

.el-tabs >>> .el-tabs__item{
  font-size: 20px;
}

.tab-body{
  margin-top: 40px;
}

.module{
  position: relative;
  margin-bottom: 60px;
}

.mod-title h2{
  display: inline-block;
  margin: 0;
  padding: 0;
  font-weight: 400;
  font-size: 20px;
  color: #0299f6;
  line-height: 18px;
}

.mod-title h2:before {
  float: left;
  content: "";
  display: inline-block;
  width: 4px;
  height: 18px;
  margin-right: 6px;
  background-color: #333;
}

.mod-content{
  margin-top: 20px;
  color: #333;
}

.mod-content .dra{
  font-size: 16px;
  line-height: 26px;
}

.more{
  float: right;
  cursor: pointer;
  font-size: 14px;
  color: #999;
  padding-right: 14px;
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAOCAYAAAASVl2WAAAABmJLR0QA/wD/AP+gvaeTAAAAv0lEQVQY013RTUpDQRAE4G8eghcR8ScgKCIugpJFjuIjqAvBc7jxj0muEnCjiIQQJOImB3GnbnpkfL1qpqqrunpSzvkDPxjhGdq2VarBF3q4wRHknP8RzvCEQzzguCalaHZwiwHecY6XogCf8TjFHh7Rh9Tx3AylIZa4TgWpSBuY4BSrYlFXKsr4bjrTW5HkJJa9SBW4jbtukmKxG5MDLOKqfzEPcB9LzQN8LSdfwxj7eMMlZvV/NFiPzFddEH4Bt5Y1mf3fnDwAAAAASUVORK5CYII=) no-repeat 100%;
}

.portrait{
  margin-bottom: 6px;
  width: 128px;
  height: 170px;
  overflow: hidden;
}

.portrait .default-img{
  width: 128px;
  height: 170px;
}

.pictures-list{
  display: flex;
}

.pictures-list>.el-image:first-child{
  width: 465px;
  height: 258px;
}

.pictures-list .default-img{
  border-style: none;
  cursor: pointer;
  width: 126px;
  height: 126px;
}

.little-pictures{
  width: 262px;
  height: 262px;
  display: flex;
  flex-wrap: wrap;
  margin-left: 10px;
  justify-content: space-between;
}

#pane-pictures .little-pictures{
  width: 100%;
  justify-content: flex-start;
  margin-left: 0;
}

#pane-pictures .default-img{
  margin-top: 10px;
  margin-left: 10px;
}

.user-avatar{
  width: 50px;
  height: 50px;
  margin-right: 20px;
}


.main-header{
  font-size: 16px;
}

.user-name{
  margin-top: 2px;
}

</style>
