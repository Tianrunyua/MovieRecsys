<template>
  <div>
    <el-menu
        :default-active="$route.path"
        class="el-menu-demo"
        mode="horizontal"
        background-color="#545c64"
        text-color="#fff"
        active-text-color="#ffd04b"
        router>
    </el-menu>
    <router-view></router-view>
  </div>
</template>



<script>
export default {
  data(){
    return {
    }
  },

  methods:{
  },
}
</script>



<style scoped>
.el-menu{
  display: flex;
  justify-content: space-between;
}

.el-menu-item{
  font-size: 16px;
}
</style>
