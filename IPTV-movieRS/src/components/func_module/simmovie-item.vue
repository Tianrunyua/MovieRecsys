<template>
  <!-- 一部电影 -->
  <div class="movie-item">
    <div v-on:mouseover="isHover = true" v-on:mouseleave="isHover = false">

      <!-- 海报 -->
      <!-- a :href="movieInfoUrl" -->
      <a @click="gotomovie(movieInfoUrl)">
        <img class="poster" :src='"../../assets/pic/poster/"+movieItem.movie_id+".jpg"'>
         <i v-if="movieItem.iswatched" style=" width: 10px;height: 10px;position: relative; left: -20%; top:-120px;
        font-weight: bolder;color: #f7ff00; background-color: #dddddd" class="el-icon-star-on ">已观看</i>
      </a>

      <!-- 鼠标划过时的效果 -->
      <div class="movie-item-hover" v-if="isHover">

        <!-- 海报 -->
        <!-- a :href="movieInfoUrl" -->
        <a @click="gotomovie(movieInfoUrl)">
          <img class="poster-hover" :src='"../../assets/pic/poster/"+movieItem.movie_id+".jpg"'>

          <!-- 鼠标划过时显示的信息 -->
          <div class="movie-hover-info">

            <div class="title-hover">
              <span class="name-tags-title">{{movieItem.movieName}}</span>
            </div>

            <div class="title-hover">
              <span class="name-tags-text">类型: </span>{{movieItem.tag}}
            </div>

            <div class="title-hover">
              <span class="name-tags-text" >相似度: </span>
            </div>

            <div class="title-hover">
              <el-progress :text-inside="true" :stroke-width="16" :percentage="movieItem.simval"></el-progress>
            </div>

          </div>
        </a>

      </div>
    </div>

    <!-- 标题 -->
    <div class="title-style">
      <!-- a :href="movieInfoUrl" -->
      <a @click="gotomovie(movieInfoUrl)">
        {{movieItem.movieName}}
      </a>
    </div>
  </div>
</template>

<script>
import moment from 'moment'
export default {
  name: "simmovie-item",
  props: {
    movieItem: Object
  },
  data() {
    return {
      circlewidth: 50,
      isHover: false,
      movieInfoUrl: ''
    }
  },
  created(){
    // 获取后端返回的数据
    this.movieInfoUrl = '/movieInfo/' + this.movieItem.movie_id
  },
  watch: {
    'movieItem'() {
      // movieItem变化时出发此函数
      this.movieInfoUrl = '/movieInfo/' + this.movieItem.movie_id
    }
  },
  methods: {
    gotomovie :function (opt){
      this.$router.push(opt)
    }
  }
}
</script>

<style scoped>
.movie-item{
  margin-top: 30px;
  margin-left: 30px;
  display: inline-block;
  vertical-align: top;
  position: relative;
}

.poster{
  /*width: 160px;*/
  width: 120px;
  height: 180px;
  /*height: 220px;*/
}

.title-style{
  /*width: 160px;*/
  width: 120px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
  font-size: 12px;
  /*font-size: 16px;*/
  color: #333;
  margin-top: 5px;
}

.score{
  color: #ffb400;
  font-size: 24px;
  font-style: italic;
}

a{
  text-decoration: none;
  color: #333;
}

.movie-item-hover{
  position: absolute;
  width: 218px;
  height: 300px;
  z-index: 10;
  top: -40px;
  left: -29px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 0 16px #fff, 0 0 6px rgba(0,0,0,.2);
}

.poster-hover{
  width: 218px;
  height: 300px;
}

.movie-hover-info{
  padding: 16px;
  height: 150px;
  box-sizing: border-box;
  position: absolute;
  z-index: 11;
  top: 150px;
  background: #fff;
  width: 218px;
  color: #999;
}

.title-hover:first-child{
  font-size: 20px;
  /*font-size: 10px;*/
  line-height: 20px;
  margin-top: 6px;
}

.title-hover{
  font-size: 16px;
  line-height: 16px;
  margin-top: 12px;
  width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.name-tags{
  font-size: 12px;
  color: #333;
}
.name-tags-title{
  font-weight: bold;
  font-size: 18px;
  color: #333;
}
.name-tags-text{
  font-weight: bold;
  font-size: 16px;
  color: #333;
}
</style>
