<template>
<div class="user-info-container">

  <div style="width: 30%">
    <h1>用户头像</h1>
     <img style="width: 120px; height:140px;"  src="../../assets/pic/user_touxiang.jpg"  >
  </div>

  <div style="width: 90%">
    <el-form :model="loginUser"  :rules="loginUserRules" ref="editFormRef" label-width="80px">

      <el-form-item label="用户名" prop="userName">
        <el-input v-model="loginUser.userName"></el-input>
      </el-form-item>

      <el-form-item label="电子邮箱" prop="email">
        <el-input v-model="loginUser.email"></el-input>
      </el-form-item>

      <el-form-item label="手机号码" prop="phoneNumber">
        <el-input v-model="loginUser.phoneNumber"></el-input>
      </el-form-item>

      <el-form-item label="性别" prop="sex">
        <el-radio-group v-model="loginUser.sex">
          <el-radio v-model="loginUser.sex" :label="'0'">男</el-radio>
          <el-radio v-model="loginUser.sex" :label="'1'">女</el-radio>
        </el-radio-group>
      </el-form-item>

        <el-button type="primary" @click="editUser">保存</el-button>
    </el-form>
  </div>

</div>
</template>



<script>
export default {
  name: "UserInfo",

  data() {
    let checkEmail = (rule, value, cb) => {
      const regEmail = /^([a-zA-Z0-9_-])+@([a-zA-Z0-9_-])+(\.[a-zA-Z0-9_-])+/
      if (regEmail.test(value)){
        return cb()
      }
      cb(new Error('请输入合法的邮箱'))
    }
    let checkMobile = (rule, value, cb) => {
      const regMobile = /^(0|86|17951)?(13[0-9]|15[0123456789]|17[678]|18[0-9]|14[57])[0-9]{8}$/
      if (regMobile.test(value)){
        return cb()
      }
      cb(new Error('请输入合法的手机号码'))
    }
    return {
      loginUser:{},

      //验证表单规则对象
      loginUserRules: {
        userName: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入电子邮箱', trigger: 'blur' },
          { validator: checkEmail, trigger: 'blur'}
        ],
        phoneNumber: [
          { required: true, message: '请输入手机号码', trigger: 'blur' },
          { validator: checkMobile, trigger: 'blur'}
        ],
        sex: [
          { required: true, message: '请选择性别', trigger: 'change' }
        ]
      }
    }
  },

  created() {
    this.getUser()
  },

  methods:{
    // 获取用户信息
    async getUser(){
      let queryInfo1 = {
        username: window.sessionStorage.getItem("loginUser")
      }
      const { data : res } = await this.$axios.get('sysMovie/getUserInfo/',{params:queryInfo1})
      this.loginUser = res.data
      console.log(this.loginUser)
    },

    async editUser(){
       let success = true
       this.$axios.defaults.headers.put['Content-Type'] = 'application/json'
       const { data: res} = await this.$axios.post('sysMovie/editUser/', JSON.stringify(this.loginUser))
       console.log(res)
       console.log(res.code)
       if (res.code !== 200){
           this.$message.error('修改用户信息失败！')
           success = false
         }
       if(!success) return
       this.$message.success('修改用户信息成功!')
       await this.getUser()
    },

  }
}
</script>


<style scoped>
.user-info-container{
  display: flex;
}
</style>

