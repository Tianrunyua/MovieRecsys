<template>
  <el-container>
    <!-- 定义页面头部区域 -->
    <el-header>
      <div class="header-inner">

        <!-- Logo区域，点击可触发toWelcome方法 -->
        <a @click="toWelcome" class="logo">
          <img style="width: 200px; height:80px;" src="../assets/pic/cuchome.jpg"  >
        </a>

        <!-- 水平模式的导航菜单，使用Element UI的el-menu组件 -->
        <el-menu
            :default-active="activeUrl"
            class="nav-menu"
            mode="horizontal"
            :router="true"
            active-text-color="#409EFF"
            text-color="#000000">

          <!-- 循环渲染菜单项，使用Element UI的el-menu-item组件 -->
          <el-menu-item :index="item.path" v-for="item in menuList" :key="item.id">
            {{item.name}}     <!-- 显示菜单项的名称 -->
           </el-menu-item>
        </el-menu>

        <!-- 搜索框容器 -->
        <div class="searchContainer">
          <el-input v-model="kw" class="searchBar" placeholder="搜索电影"></el-input>
          <el-button id="searchBtn" icon="el-icon-search" type="primary" circle @click="search"></el-button>
        </div>

        <!-- 下拉菜单，使用Element UI的el-dropdown组件。@command="handleCommand"是一个事件监听器，它监听用户选择下拉菜单中的某个命令时触发的事件。 -->
        <el-dropdown @command="handleCommand">

          <span class="el-dropdown-link">
            <!-- 用户头像，使用Element UI的el-avatar组件 -->
            <el-avatar :src="url" :size="50">{{url === '' || url === null ? 'user': ''}}</el-avatar>
            <!-- 下拉箭头图标 -->
            <i class="el-icon-arrow-down el-icon--right icon-arrow"></i>
          </span>

          <!-- 下拉菜单内容，使用Element UI的el-dropdown-menu组件 -->
          <el-dropdown-menu slot="dropdown">
            <!-- 个人中心下拉菜单项，使用Element UI的el-dropdown-item组件 -->
            <el-dropdown-item command="userMenu" v-if="isToken">个人中心</el-dropdown-item>
            <!-- 退出下拉菜单项 -->
            <el-dropdown-item command="logout" v-if="isToken">退出</el-dropdown-item>
            <!-- 登录下拉菜单项 -->
            <el-dropdown-item command="login" v-if="!isToken">登录</el-dropdown-item>

          </el-dropdown-menu>
        </el-dropdown>

      </div>
    </el-header>

    <!-- 主要内容区域，使用Element UI的el-main组件 -->
    <el-main>
      <router-view></router-view>
    </el-main>

    <!-- 定义页面底部区域 -->
    <el-footer>
      <div class="footer-mini"></div>
      <div class="footer"></div>
      <el-backtop></el-backtop>    <!-- 返回顶部按钮，使用Element UI的el-backtop组件 -->
    </el-footer>

  </el-container>
</template>

<script>
export default {
  name: "Welcome",
  data() {
    return {
      isToken: '',       // 用于存储用户的登录令牌，初始化为空字符串
      url: '',           // 用于存储用户头像的 URL

      // 用于存储当前激活的菜单项的路径（高亮？），见水平模式的导航菜单
      activeUrl: this.$route.path.substring(0, this.$route.path.indexOf('/',1) === -1 ? this.$route.path.length : this.$route.path.indexOf('/',1)),
      menuList: [
        { id: 1, name: '首页', path: '/home' },
        { id: 2, name: '电影', path: '/movie' },
        { id: 3, name: '推荐', path: '/Recommend' },
        { id: 4, name: '历史', path: '/history' }
      ],

      kw: ''   // 用于绑定搜索框的关键字数据
    }
  },

  created() {  // Vue 实例的生命周期钩子，当实例被创建之后被调用
    this.isToken = window.sessionStorage.getItem("token")  // 从浏览器的 sessionStorage 中获取 "token" 项，并赋值给 isToken
  },

  watch: {
    '$route'() {    // 监听路由对象 $route 的变化
      this.activeUrl = this.$route.path.substring(0, this.$route.path.indexOf('/',1) === -1 ? this.$route.path.length : this.$route.path.indexOf('/',1))
    }
  },

  methods: {
    search() {
      let tmp = this.kw
      this.$router.push('/search/searchMovie?kw=' + tmp)
      this.kw = ''
    },

    handleCommand(command) {     // 接收一个 command 参数
      if(command === 'logout') {
        let queryinfo = {
           username: window.sessionStorage.getItem("loginUser"),
        }
        const {data: res} =  this.$axios.get('sysMovie/quit/', {params:queryinfo});
        return this.$router.push('/login')  //使用 Vue Router 的 push 方法导航到登录页面。
      }
      this.$router.push('/' + command)
    },

    toWelcome() {
      this.$router.push('/welcome')
    }
  }

}
</script>

<style scoped>
.el-header{
  height: 80px !important;
  border-bottom: 1px solid #e6e6e6;
}

.header-inner{
  width: 75%;
  margin: 0 12.5%;
  display: flex;
}

.logo{
  height: 80px;
  width: 200px;
}

.nav-menu{
  border-bottom: 0px solid #eee !important;
  height: 60px;
  margin: 9px auto;
}

.el-menu-item{
  font-size: 20px;
}

.el-menu-item:hover{
  color: #409EFF !important;
}

.searchContainer{
  padding: 20px 0;
  display: flex;
}

.searchBar >>> input{
  border-radius: 50px;
}

#searchBtn{
  transform: translate(-100%, 0);
}

.el-dropdown-link{
  width: 70px;
  margin: 15px 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.icon-arrow {
  transition: all 0.2s ease-in-out;
  -webkit-transition: all 0.2s ease-in-out;
  -moz-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
}

.el-dropdown-link:hover .icon-arrow{
  /* 定义动画的状态 */
  -webkit-transform: rotate(180deg) scale(1);
  -moz-transform: rotate(180deg) scale(1);
  -o-transform: rotate(180deg) scale(1);
  -transform: rotate(180deg) scale(1);
}

.footer-mini{
  border-top: 1px solid #EEE;
  padding-top: 20px;
  text-align: center;
}

.footer{
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.el-main{
  padding: 0px;
}
</style>
