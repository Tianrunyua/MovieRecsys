<template>
  <div>

    <!-- 创建一个搜索容器 -->
    <div class="searchContainer">
      <el-input v-model="keyword" class="searchBar" placeholder="请输入电影名称"></el-input>
      <el-button id="searchBtn" icon="el-icon-search" type="primary" circle @click="searchInfo"></el-button>
    </div>

    <!-- 创建一个导航菜单容器 -->
    <div>
      <!-- default-active绑定到当前路由的路径，以实现菜单项的激活状态 -->
      <el-menu :default-active="$route.path"
               class="nav-menu"
               mode="horizontal"
               :router="true"
               background-color="#47464a"
               text-color="#fff"
               active-text-color="#ffd04b">
        <el-menu-item :index="item.path" v-for="item in menuList" :key="item.id">{{item.name}}</el-menu-item>
      </el-menu>
    </div>

    <!-- router-view组件用于渲染Vue Router匹配到的组件 -->
    <!-- 将keyword作为prop传递给渲染的组件 -->
    <div>
      <router-view :keyword="keyword"></router-view>
    </div>
  </div>


</template>





<script>
export default {
  name: "Search",
  data() {
    return {
      keyword: this.$route.query.kw,
      menuList: [
        {
          id: '1',
          name: '电影',
          path: '/search/searchMovie'
        }
      ]
    }
  },

  created() {

  },

  methods:{
    searchInfo(){
      this.$router.push(this.$route.path + '?kw=' + this.keyword)
    }
  },

  watch:{
    '$route'(){
      this.keyword = this.$route.query.kw
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
  font-size: 16px;
}

.el-menu-item{
  font-size: 16px;
}

.el-menu-item:hover{
  color: #FFFFFF !important;
}

.searchContainer{
  padding: 30px 0;
  display: flex;
  background: #47464a;
  justify-content: center;
}

.searchBar{
  width: 630px;
}

.searchBar >>> input{
  padding-right: 0;
  border-radius: 50px;
  width: 630px;
  height: 50px;
}

#searchBtn{
  transform: translate(-100%, 0);
  width: 50px;
  height: 50px;
}
</style>
