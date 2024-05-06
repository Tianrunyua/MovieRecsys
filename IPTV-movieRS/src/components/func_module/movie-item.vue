<template>
  <!-- 一部电影 -->
  <div class="movie-item">
    <div v-on:mouseover="isHover = true" v-on:mouseleave="isHover = false">

      <!-- 海报 -->
      <!-- a :href="movieInfoUrl" -->
      <a @click="gotomovie(movieInfoUrl)">
        <img class="poster" :src='"../../assets/pic/poster/"+movieItem.movie_id+".jpg"'>
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
              <span class="name-tags">{{movieItem.movieName}}</span>
            </div>
            <div class="title-hover">
              <span class="name-tags">类型: </span>{{movieItem.tag}}
            </div>
            <div class="title-hover">
              <span class="name-tags">上映时间: </span>{{movieItem.releaseDate}}
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
  name: "movie-item",
  props: {
    movieItem: Object  //props 是子组件可以接收的属性，它们是动态的，可以从父组件传入，并且可以是任何类型的数据（如字符串、数字、对象、数组等）。
  },
  data() {
    return {
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
      // movieItem变化时，会运行这个方法
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
  width: 160px;
  height: 220px;
}

.title-style{
  width: 160px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
  font-size: 16px;
  color: #333;
  margin-top: 10px;
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
  color: #333;
}
</style>
