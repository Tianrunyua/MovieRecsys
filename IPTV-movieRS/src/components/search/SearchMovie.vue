<template>
  <div class="search-container">

    <!-- 搜索的一部电影 -->
    <div class="search-result" v-for="item in movieList" :key="item.movie_id">

      <!-- 电影海报 -->
      <div class="img-container">
        <a :href="'/movieInfo/' + item.movie_id">
          <img :src='"../../assets/pic/poster/"+item.movie_id+".jpg"'>
        </a>
      </div>

      <!-- 电影信息 -->
      <div class="movie-info">
        <div class="movie-name">{{item.movieName}}</div>
        <br/>
        <div class="movie-category">{{item.tag}}</div>
        <br/>
        <div class="movie-releaseDate">{{item.releaseDate}}上映</div>
      </div>

    </div>

  </div>
</template>




<script>
export default {
  name: "SearchMovie",

  props:{
    keyword: {
      type: String,
      default: ''
    }
  },

  data() {
    return{
      movieList: [
        {
          tag: '',
          movie_id: 0,
          movieLength: 0,
          movieName: '',
          releaseDate: ''
        }
      ]
    }
  },

  created() {
    this.getSearchMovieList()
  },

  methods:{
    async getSearchMovieList(){
      let queryInfo = {
        movieName: this.keyword
      }
      const { data : res } = await this.$axios.get('sysMovie/find', {params : queryInfo})
      console.log(res)
      if(res.code !== 1) return this.$message.error('获取信息失败')
      this.movieList = res.data
      //处理数据
    }
  },
  watch:{
    '$route'(){
      this.getSearchMovieList()
    }
  }
}
</script>




<style scoped>
.search-container{
  width: 980px;
  margin: 0px auto;
  margin-bottom: 20px;
  display: flex;
  flex-wrap: wrap;
}

.search-result{
  width: 460px;
  margin: 30px 0 0 30px;
  display: flex;
}

.img-container{
  width: 160px;
  height: 220px;
  margin-right: 20px;
}

.img-container img{
  width: 100%;
  height: 100%;
}

a{
  text-decoration: none;
  cursor:pointer;
}

.movie-info{
  display: flex;
  flex-direction: column;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  margin-top: 5px;
  font-size: 12px;
  width: 250px;
  border-bottom: solid #eee 1px;
}

.movie-name{
  font-size: 18px;
  margin-top: 27px;
}

.movie-ename{
  font-size: 14px;
  color: #999;
  margin-top: 3px;
}

.movie-score{
  font-size: 26px;
  margin-bottom: 13px;
  color: #ffb400;
  height: 31px;
  line-height: 31px;
}

.movie-category{
  color: #333;

}

.movie-releaseDate{
  color: #999;
}
</style>
