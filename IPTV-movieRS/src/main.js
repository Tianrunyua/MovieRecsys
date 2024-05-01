// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from 'axios'
import global_variable from "@/components/global_variable"

Vue.prototype.global = global_variable
Vue.config.productionTip = false

axios.defaults.baseURL = 'http://localhost:8000/'
axios.interceptors.request.use(config => {
  // config.headers.Token = window.sessionStorage.getItem('token')
  return config
})

Vue.use(ElementUI)
Vue.prototype.$axios = axios

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
