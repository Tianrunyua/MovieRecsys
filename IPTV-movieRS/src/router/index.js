import Vue from 'vue'
import Router from 'vue-router'
import VueRouter from 'vue-router'
import HelloWorld from '@/components/HelloWorld'

import Login from '@/components/Login'
import Welcome from '@/components/Welcome'

import Home from "@/components/Home";
import Movie from "@/components/Movie";
//import history from "@/components/History";

import RecommendMain from "@/components/Recommend/RecommendMain";
import RecommendOne from "@/components/Recommend/Recommendone";

import UserMenu from "@/components/user/UserMenu";
import UserInfo from "@/components/user/UserInfo";

import Search from "@/components/search/Search";
import SearchMovie from "@/components/search/SearchMovie";

import MovieInfo from "@/components/MovieInfo";

Vue.use(Router)
//Vue.prototype.$base_url = 'http://localhost:8000'

export default new Router({
  routes: [
//    {
//      path: '/',
//      name: 'HelloWorld',
//      component: HelloWorld
//    }
    {
      path:'/',
      redirect: { name: 'Login_page'}
    },
    {
      path: '/login',
      name: 'Login_page',
      component: Login
    },
    {
      path: '/welcome',
      name: 'Welcome_page',
      component: Welcome,

      redirect: { name: 'home'},

      children: [
        {
          path: '/home/',
          name: 'home',
          component: Home
        },
        {
          path: '/movie',
          name: 'movie',
          component: Movie
        },
//        {
//          path: '/history/',
//          name: 'aboutus',
//          component: history
//        },
        {
          path: '/Recommend/',
          name: 'recommendmain',
          component: RecommendMain,
//          redirect: '/Recommend/RSDemo',
//          children:[
//            {
//              path: 'RSDemo',         // 三级 不用/  二级后面带上/就可以
//              name: 'recommendone',
//              RecommendOne
//            }
//          ]
        },
        {
          path: '/userMenu',         // 起导航栏的作用
          name: 'userMenu',
          component: UserMenu,
          redirect: {name: 'userInfo'},   // 导航栏+userInfo界面内容
          children:[
            {
              path: '/userinfo',       // 三级 不用/  二级后面带上/就可以
              name: 'userInfo',
              component: UserInfo,
            }
          ]
        },
        {
          path: '/search/',
          name: 'search',
          component: Search,
          redirect: { name:'searchMovie' },
          children: [
            {
              path:'searchMovie',
              name: 'searchMovie',
              component: SearchMovie,
            }
          ]
        },
        {
          path: '/movieInfo/:movie_id',
          name: 'movieInfo',
          component: MovieInfo
        },
      ]
    },
  ]
})


const originalPush = VueRouter.prototype.push

VueRouter.prototype.push = function push(location) {
  return originalPush.call(this, location).catch(err => err)
}
