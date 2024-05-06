import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'

import Login from '@/components/Login'
import Welcome from '@/components/Welcome'

import Home from "@/components/Home";
import Movie from "@/components/Movie";
import history from "@/components/History";
import Rcommend from "@/components/Recommend";

import UserMenu from "@/components/user/UserMenu";

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
        {
          path: '/history/',
          name: 'aboutus',
          component: history
        },
        {
          path: '/Recommend/',
          name: 'recommendList',
          component: Rcommend
        },
        {
          path: '/userMenu',
          name: 'userMenu',
          component: UserMenu
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


