<template>

  <div class="login_container">

    <!-- 条件渲染，如果showtag为false，则显示以下内容 -->
    <div v-if="!showtag" class="login_box">

      <!-- 登录表单的标题区域 -->
      <div class="title_box">
        <p>用户登录</p>
      </div>

      <!-- 登录表单区域 -->
      <el-form class="login_form" :model="loginForm" :rules="loginFormRules" ref="loginFormRef">

        <!-- 用户名输入框 -->
        <el-form-item prop="userName">
          <el-input v-model="loginForm.userName" placeholder="请输入用户名"></el-input>
        </el-form-item>

        <!-- 密码输入框 -->
        <el-form-item prop="password">
          <el-input type="password" v-model="loginForm.password" placeholder="请输入密码"></el-input>
        </el-form-item>

        <!-- 登录按钮区域 -->
        <el-form-item class="btns">
          <el-button type="primary" size="medium" :round="true" @click="showtagpage">点击登录</el-button>
        </el-form-item>

      </el-form>

    </div>


    <!-- 条件渲染，如果showtag为true，则显示以下内容 -->
    <div v-if="showtag" class="loginstep2" style="position:relative;left:38%;top:30%">
        <h3>选择标签</h3>

        <!-- 标签选择区域，使用v-for循环显示所有标签 -->
        <div class="alltag">
          <span class="tagbox" v-for="(item,index) in showtags" :key="index">
            <input type="checkbox" v-model="select_tags" name="tagbox" :value="item" />
            <b>{{item}}</b>
          </span>
        </div>

        <!-- 按钮区域，包含“跳过”和“进入系统”两个按钮 -->
        <div class="twobtn">
          <button class="skip" @click="login('skip')">跳过</button>
          <button class="go" @click="login">进入系统</button>
        </div>
      </div>

  </div>
</template>

<script>

export default {
  name: "Login",
  data() {
    return {
      // 登录表单数据对象
      loginForm:{
        userName : 'User53',
        password : '123456'
      },
      // 登录表单验证规则
      loginFormRules: {
        // 验证用户名格式
        userName: [
          { required: true, message: "请输入用户名称", trigger: "blur"},
          { min:2, max: 20, message: "用户名称长度在2到20个字符之间", trigger: "blur"}
        ],
        // 验证密码格式
        password: [
          { required: true, message: "请输入密码", trigger: "blur"},
          { min:6, max: 16, message: "登录密码长度在6到16个字符之间", trigger: "blur"}
        ]
      },
      sessionId: 0,
      showtag: false,
      showtags :['剧情', '喜剧', '动作', '爱情', '冒险', '悬疑', '惊悚', '家庭', '犯罪', '动画',
        '感人', '励志', '搞笑', '科幻', '奇幻', '经典', '历史', '传记', '战争', '恐怖', '青春', '亲情', '儿童',
        '文艺', '古装', '亲子', '武侠', '警匪', '灾难'],
      select_tags:[],
    }
  },
  created() {
    this.sessionId = window.sessionStorage.getItem('sessionId')
    console.log('this sessionId is : ' + this.sessionId)
    window.sessionStorage.setItem('sessionId', 0)
  },
  methods: {
    // success(params) {
    //  this.login()
    // },

    showtagpage() {

      // 验证登录表单的有效性
      this.$refs.loginFormRef.validate(async valid => {
        if(!valid) return;

        // 设置axios请求头，指定内容类型为JSON
        this.$axios.defaults.headers.post['Content-Type'] = 'application/json'

        // 发送登录请求到后端接口
        const {data: res} = await this.$axios.post('/sysMovie/login', JSON.stringify(this.loginForm));
        // 如果登录失败，则显示错误信息
        if(res.code !== 200) return this.$message.error(res.msg);
        // 显示登录成功的提示信息
        this.$message.success({message: "登录成功", duration: 1000});
        console.log(res.data);

        // 保存后端返回的token和用户信息到会话存储
        console.log(res.data.token);
        console.log(res.data.sysUser);
        window.sessionStorage.setItem("token", res.data.token);
        window.sessionStorage.setItem("loginUser", res.data.sysUser);

        // 显示标签选择界面
        this.showtag = true
      })
    },

    // 处理在标签选择页 跳过 或 进入系统 的方法
    login :function (opt) {
      // 如果用户选择跳过，则默认选择一些标签
      if (opt && opt === 'skip') {
        this.usertags = ['剧情', '喜剧', '爱情','动作']
      } else {
        // 如果用户选择的标签少于4个，则提示用户至少选择4个标签
        if (this.select_tags.length < 4 && this.select_tags.length >= 0) {
          this.$alert('至少选择4个标签', '温馨提示', {
            confirmButtonText: 'OK'})
          return false
        } else {
          // 标签足够，将用户选择的标签保存到usertags
          this.usertags = this.select_tags
        }
      }
      // 将选择4个的标签保存到会话存储 / 设置本地标签数据
      window.sessionStorage.setItem('homostep',0)
      window.sessionStorage.setItem('default_tag','全部')
      window.sessionStorage.setItem('tag1', this.usertags[0])
      window.sessionStorage.setItem('tag2', this.usertags[1])
      window.sessionStorage.setItem('tag3', this.usertags[2])
      window.sessionStorage.setItem('tag4', this.usertags[3])
      // console.log(window.sessionStorage.getItem('tag4'))
      // 根据选择标签导航跳转到首页，首页面需要接收参数，或者设置本地参数
      this.$router.push('/welcome');
    },

    // registerAccount() {
    //   this.$router.push('/register')
    //}
  }
}
</script>

<style lang="less" scoped>
.login_container{
  background: url("../assets/pic/bg.jpg");
  background-size: 100% 100%;
  height: 100%;
  width: 100%;
  position: fixed;
}

.login_box{
  width: 500px;
  height: 350px;
  background-color: #fff;
  border-radius: 3px;
  position: absolute;
  left: 52%;
  top: 60%;
  transform: translate(-50%, -50%);
}

.avatar_box{
  height: 130px;
  width: 130px;
  border: 1px solid #eee;
  border-radius: 50%;
  padding: 10px;
  box-shadow: 0 0 10px #ddd;
  position: absolute;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: #fff;
}

.avatar_box > img{
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background-color: #eee;
}

.title_box{
  text-align: center;
  font-size: 200%;
}

.login_form{
  position: absolute;
  bottom: 0;
  width: 100%;
  padding: 0 20px;
  box-sizing: border-box;
}

.btns{
  display: flex;
  justify-content: center;
}
#logins(){
    //width: 40%;
    width: 500px;
    background-color: rgba(234, 244, 252, 0.83);
    border-radius:8px;
    min-height: 300px;
    display: flex;
    flex-flow: column;
    justify-content: center;
    align-items: center;
    border: 1px solid #eee;
    padding-bottom: 20px;
    margin-bottom: 20px;
  }
#btns(){
    background: #fff;
    height:35px;
    line-height: 35px;
    border:1px solid #999;
    border-radius: 5px;
    margin-top:10px;
    cursor: pointer;
    outline: none;
  }
@baseColor: #408afa;
.loginstep2{
      #logins();
      justify-content: space-around;
      h3{
        margin-top:10px;
      }
      .twobtn{
        width: 100%;
        display: flex;
        justify-content:space-around;
        .skip,.go{
          width: 30%;
          #btns();
          &:hover{
            color: #fff;
            background: @baseColor;
          }
        }

        .skip:hover{
          background: #999;
        }
      }
      .alltag{
        padding: 5px;
        .tagbox{
          display: inline-flex;
          padding: 4px;
          border: 1px solid #999;
          border-radius: 3px;
          justify-content: center;
          align-items: center;
          margin:8px;
          input{
            cursor: pointer;
          }
        }
      }
    }
</style>
