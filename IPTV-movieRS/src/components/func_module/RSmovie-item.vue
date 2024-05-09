<template>
  <el-container style="height: 230px; border: 2px solid #eee; box-shadow: 0 0 1px 1px #ddd; ">

    <!-- 海报 -->
    <el-aside width="200px" height="160px">
        <!-- a :href="movieInfoUrl" -->
        <a @click="gotomovie(movieInfoUrl)">
          <img class="poster" :src='"../../assets/pic/poster/"+movieItem.movie_id+".jpg"'>
        </a>
    </el-aside>

    <!-- 推荐电影的基本信息 -->
    <el-main height="160px">
      <el-row>
        <el-col :span="6"><div class="grid-content-up" style="font-weight: bolder">{{movieItem.movieName}}</div></el-col>
        <el-col :span="6"><div class="grid-content-up text-oneLine" style="font-weight: bolder" >{{movieItem.author}}</div></el-col>
        <el-col :span="6"><div class="grid-content-up"  style="overflow: visible;font-weight: bolder;">{{movieItem.tag}}</div></el-col>
        <el-col :span="4"><div class="grid-content-up"  style="overflow: visible;font-weight: bolder; ">{{movieItem.area}}</div></el-col>
      </el-row>

      <el-row>
        <!-- 相似度和共现度 -->
        <el-col :span="12" >
          <el-row >
            <el-col :span="4" ><div class="grid-content-down3  " style="text-align: center"><p style="font-weight: bolder;margin-top: 25px" >视觉模态:</p></div></el-col>
            <el-col :span="20" ><div class="grid-content-down3  "><p style="margin-top: 25px">{{movieItem.simv}}</p></div></el-col>
          </el-row>
          <el-row >
            <el-col :span="4" ><div class="grid-content-down3  " style="text-align: center"><p style="font-weight: bolder">文本模态:</p></div></el-col>
            <el-col :span="20" ><div class="grid-content-down3  "><p>{{movieItem.simt}}</p></div></el-col>
          </el-row>
          <el-row >
            <el-col :span="4" ><div class="grid-content-down3  " style="text-align: center"><p style="font-weight: bolder">共现度:</p></div></el-col>
            <el-col :span="20" ><div class="grid-content-down3  "><p>{{movieItem.com}}</p></div></el-col>
          </el-row>
        </el-col>

        <!-- 理由和兴趣值 -->
        <el-col :span="8">
          <el-row>
            <div class="grid-content-down1 ">
              <p style="font-weight: bolder;margin-top: 30px">推荐理由:</p>{{movieItem.reason}}
            </div>
          </el-row>
          <el-row>
            <el-col :span="4" ><div class="grid-content-down3  " style="text-align: left"><p style="font-weight: bolder">兴趣值:</p></div></el-col>
            <el-col :span="20" ><div class="grid-content-down3  "><p>{{movieItem.score}}</p></div></el-col>
          </el-row>
        </el-col>

        <!-- 偏好值 -->
        <el-col :span="4">
          <el-row >
            <el-col :span="8"><div class="grid-content-down2 "><p style="font-weight: bolder;margin-top: 25px">视觉偏好:</p></div></el-col>
            <el-col :span="16"><div class="grid-content-down2 " ><el-progress style="margin-top: 10px" type="circle" color="#67C23A" :width=40  :percentage="movieItem.v_att" ></el-progress></div></el-col>
          </el-row>
          <el-row >
            <el-col :span="8"><div class="grid-content-down2 "><p style="font-weight: bolder;margin-top: 25px">文本偏好:</p></div></el-col>
            <el-col :span="16"><div class="grid-content-down2 "><el-progress style="margin-top: 10px"type="circle" color="#E6A23C" :width=40  :percentage="movieItem.t_att" ></el-progress></div></el-col>
          </el-row>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>



<script>
import moment from 'moment'
export default {
  name: "movie-item",
  props: {
    movieItem: Object,
  },
  data() {
    return {
      isHover: false,
      movieInfoUrl: ''
    }
  },
  created(){
    this.movieInfoUrl = '/movieInfo/' + this.movieItem.movie_id
  },
  methods: {
    gotomovie :function (opt){
      this.$router.push(opt)
    }
  },
  watch: {
    'movieItem'() {
      this.movieInfoUrl = '/movieInfo/' + this.movieItem.movie_id
    }
  }
}
</script>





<style lang="less" scoped>

.poster{
  width: 150px;
  height: 190px;
  margin-top: 15px;
  //margin-left: 5px;
  align-content: center;
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
  width: 190px;
  height: 200px;
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
.el-aside {
    background-color: #ffffff;
    color: #333;
    text-align: center;
  }
.el-main {
    background-color: #ffffff;
    color: #333;
    border-radius: -5px
  }
.el-row {
    //margin-bottom: 0px;
    //&:last-child {
    //  margin-bottom: 0;
    //}
  }
.el-col {
    border-radius: -5px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content-up {
    //border-radius: 4px;
    text-align: center;
    min-height: 20px;
  }
  .grid-content-down1 {
    //border-radius: 4px;
    min-height: 80px;
  }
  .grid-content-down2 {
    //border-radius: 4px;
    min-height: 40px;
    text-align:center;

  }
  .grid-content-down3 {
    //border-radius: 1px;
    min-height: 30px;
  }
  .text-oneLine{
      display:block;            /*内联对象需加*/
      word-break:keep-all;      /* 不换行 */
      white-space:nowrap;       /* 不换行 */
      overflow:hidden;          /* 内容超出宽度时隐藏超出部分的内容 */
      text-overflow:ellipsis;   /* 当对象内文本溢出时显示省略标记(...) ；需与overflow:hidden;一起使用。*/
  }
</style>
