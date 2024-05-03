import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import Login from '@/components/Login'
import Welcome from '@/components/Welcome'

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
      name: 'Login_page',
      redirect: 'Login'
    },
    {
      path: '/login',
      name: 'Login_page',
      component: Login
    },
    {
      path: '/welcome',
      name: 'Welcome_page',
      component: Welcome
    },
  ]
})


