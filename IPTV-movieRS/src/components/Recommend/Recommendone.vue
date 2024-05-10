<template>
  <div class="recommonContain">哈哈哈哈哈
    <div class="mainContent">
      <div class="Con ">
        <div class="allsign">

          <!-- 推荐列表的标题行，使用无序列表ul来布局 -->
          <ul class="relists-title">
            <li class="relist title">
              <p class="rename">封面图</p>
              <p class="rename">节目名称</p>
              <p class="rename">演员</p>
              <p class="rename">类型</p>
              <p class="rename">制片地区</p>
              <!--el-button  type="primary" @click="getRSData('all')"  plain size="mini">推荐</el-button-->
            </li>
          </ul>

          <!-- 水平分隔线，使用html的hr标签，并添加了样式定义 -->
          <hr style="border: 5px dotted #ddd"/>

          <!-- 推荐列表的内容区域，使用无序列表ul -->
          <ul class="relists-contain">
            <!-- 使用v-for指令循环渲染movies数组中的每一项电影对象 -->
            <!-- 为每个循环创建一个RSmovie-item组件实例，并将当前电影对象item作为prop传递 -->
            <RSmovie-item :movieItem="item" v-for="(item, index) in movies" :key="index" ></RSmovie-item>
          </ul>

        </div>
      </div>
    </div>

    <!-- 分页助手容器 -->
    <div class="pageHelper">
      <el-pagination
          @current-change="handleCurrentChange"
          :current-page="curpageNum"
          :page-size="pageSize"
          background
          layout="prev, pager, next"
          :total="total">
      </el-pagination>
    </div>

  </div>
</template>



<script>
import RSmovieItem from '../func_module/RSmovie-item';

export default {
  name: 'recommendone',
  components: {
    RSmovieItem
  },
  data () {
    return {
      movies: {},

      // 分页相关
      total: 0, // 总条数
      curpageNum: 1, // 当前激活页
      pageSize: 5, // 每页显示多少条
      tag: 'all',
    }
  },

  methods: {
    async getRSData () {
      let queryInfo = {
        username: window.sessionStorage.getItem("loginUser"),
        pageSize: 5,
        curpageNum: this.curpageNum,
      }
      const { data : res } = await this.$axios.get('sysMovie/RS', {params: queryInfo})
      this.movies = res.data.data
      console.log(res)
      console.log(this.movies)
      this.total = res.data.total   // 用于分页
    },

    handleCurrentChange(newPage) {
      this.curpageNum = newPage
      this.getRSData ()
    }
  },

  created() {
    this.getRSData ()
  },
}
</script>




<style lang="less" scoped>
  @baseColor:#20a0ff;
  #ellies(@n){
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-line-clamp: @n;
    -webkit-box-orient: vertical;
    white-space: nowrap;
  }
  .recommonContain{
    width: 100%;
    padding:2% 8%;
    padding-bottom: 0;
    box-sizing: border-box;
    .mainContent{
      width: 100%;
      display: flex;
      box-sizing: border-box;
      justify-content: space-around;
      .Con{
        box-sizing: border-box;
        padding:10px;
        border:1px solid #ddd;
        //box-shadow: 0 0 5px 5px #eee;
        min-height: 500px;
        margin-top:15px;
        margin-bottom: 15px;
        margin-left: 10px;
        .lists{
          margin-top:10px;
          margin-left: 100%;
          li{
            display: inline-block;
            border: 1px solid #ddd;
            box-sizing: border-box;
            padding: 6px;
            border-radius: 4px;
            margin:5px;
            text-align: center;
            cursor: pointer;
            font-size: 12px;
            &:hover{
              color: @baseColor;
              border: 1px solid @baseColor;
            }
          }
          .oktag{
            color: @baseColor;
            border: 1px solid @baseColor;
          }
          .moretag{
            color: orange;
            border: 1px solid orange;
          }
        }
        .relists{
          margin-top:20px;
          display: flex;
          justify-content: space-around;
          flex-flow: row;
          .more{
            margin:auto;
            padding: 5px;
            border:1px solid orange;
            color:#fff;
            background: orange;
            text-align: center;
            width: 150px;
            border-radius: 3px;
            height: 30px;
            line-height: 30px;
            margin-top:15px;
            cursor: pointer;
          }
          .relist{
            width: 100%;
            box-sizing: border-box;
            padding: 40px;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: space-around;
            box-shadow: 0 0 1px 2px #ddd;
            /*&:hover{
              color: @baseColor;
              box-shadow: 0 0 20px 5px #ddd;
            }*/
			.bookimg:hover{
            cursor: pointer;
			  box-shadow: 0 0 20px 5px #ddd;
			}
            .recreater{
              font-size: 12px;
              color: #666;
              line-height: 14px;
              margin-bottom: 5px;
              margin-top: 5px;
              #ellies(1)
            }
            .recreater:hover {   /* 使缩略的文字可显示 */
              overflow: visible;
            }
            .rename{
              font-size: 14px;
              line-height: 16px;
              #ellies(1)
            }
            img{
              width: 100px;
              height:120px;
              border-radius: 5px;
            }
            p{
              width: 20%;
              box-sizing: border-box;
            }
            p:nth-child(1){
              padding-left: 20px;
            }
            p:nth-child(4),p:nth-child(5),p:nth-child(6){
              width: 13%;
            }
          }
          .title{
            box-shadow: none;
            margin-bottom: 30px;
            &:hover{
              box-shadow: none;
              color: #666;
              cursor: auto;
            }
          }
          .onelist{
            width: 100%;
            color:#666;
            margin:5px 0;
            display: flex;
            justify-content: space-between;
            cursor: pointer;
            &:hover{
              color:@baseColor;
            }
            .onetime,.onename{
              #ellies(1);
              display: inline-block;
              box-sizing: border-box;
              padding: 0 10px;
            }
          }
          .Rsbt{
            width: 100%;
            margin-left:100%;
            }
        }
        .relists-title{
          margin-top:10px;
          display: flex;
          //justify-content: space-around;
          justify-content: normal;
          flex-flow: column;
          .more{
            margin:auto;
            padding: 5px;
            border:1px solid orange;
            color:#fff;
            background: orange;
            text-align: center;
            width: 150px;
            border-radius: 3px;
            height: 30px;
            line-height: 30px;
            margin-top:15px;
            cursor: pointer;
          }
          .relist{
            width: 100%;
            box-sizing: border-box;
            padding:3px;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: space-around;
            box-shadow: 0 0 1px 2px #ddd;
            /*&:hover{
              color: @baseColor;
              box-shadow: 0 0 20px 5px #ddd;
            }*/
			.bookimg:hover{
            cursor: pointer;
			  box-shadow: 0 0 20px 5px #ddd;
			}
            .recreater{
              font-size: 12px;
              color: #666;
              line-height: 14px;
              margin-bottom: 5px;
              margin-top: 5px;
              #ellies(1)
            }
            .recreater:hover {   /* 使缩略的文字可显示 */
              overflow: visible;
            }
            .rename{
              font-size: 20px;
              //line-height: 20px;
              #ellies(1)
            }
            img{
              width: 100px;
              height:120px;
              border-radius: 5px;
            }
            p{
              width: 20%;
              box-sizing: border-box;
            }
            p:nth-child(1){
              padding-left: 20px;
            }
            p:nth-child(4),p:nth-child(5),p:nth-child(6){
              width: 13%;
            }
          }
          .title{
            box-shadow: none;
            margin-bottom: 30px;
            &:hover{
              box-shadow: none;
              color: #666;
              cursor: auto;
            }
          }
          .onelist{
            width: 100%;
            color:#666;
            margin:5px 0;
            display: flex;
            justify-content: space-between;
            cursor: pointer;
            &:hover{
              color:@baseColor;
            }
            .onetime,.onename{
              #ellies(1);
              display: inline-block;
              box-sizing: border-box;
              padding: 0 10px;
            }
          }
          .Rsbt{
            width: 100%;
            margin-left:100%;
            }
        }
        .relists-contain{
          margin-top:20px;
          margin-left: -20px;
          margin-right: 20px;
          display: flex;
          justify-content: center;
          flex-flow: column;
          align-content: center;
          .more{
            margin:auto;
            padding: 5px;
            border:1px solid orange;
            color:#fff;
            background: orange;
            text-align: center;
            width: 150px;
            border-radius: 3px;
            height: 30px;
            line-height: 30px;
            margin-top:15px;
            cursor: pointer;
          }
          .relist{
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: space-around;
            box-shadow: 0 0 1px 2px #ddd;
            flex-flow: row;
            flex-wrap: wrap;
              .bookimg:hover{
                      cursor: pointer;
                  box-shadow: 0 0 20px 5px #ddd;
                }
                .recreater{
                  font-size: 12px;
                  color: #666;
                  line-height: 14px;
                  margin-bottom: 5px;
                  margin-top: 5px;
                  #ellies(1)
                }
                .recreater:hover {   /* 使缩略的文字可显示 */
                  overflow: visible;
                }
                .rename{
                  font-size: 14px;
                  line-height: 16px;
                  #ellies(1)
                }
                img{
                  width: 140px;
                  height:200px;
                  border-radius: 5px;
                }
                p{
              width: 20%;
              box-sizing: border-box;
            }
                p:nth-child(1){
              padding-left: 20px;
            }
                p:nth-child(4),p:nth-child(5),p:nth-child(6){
              width: 13%;
            }


          }
          .title{
            box-shadow: none;
            margin-bottom: 30px;
            &:hover{
              box-shadow: none;
              color: #666;
              cursor: auto;
            }
          }
          .onelist{
            width: 100%;
            color:#666;
            margin:5px 0;
            display: flex;
            justify-content: space-between;
            cursor: pointer;
            &:hover{
              color:@baseColor;
            }
            .onetime,.onename{
              #ellies(1);
              display: inline-block;
              box-sizing: border-box;
              padding: 0 10px;
            }
          }
          .Rsbt{
            width: 100%;
            margin-left:100%;
            }
        }
      }
      .bookCon{
        flex: 2;
        margin-right: 15px;
      }
      .Con{
        flex: 1;
        margin-left: 15px;
        min-width: 30%;
      }
    }
    .rightpag{
      width: 100%;
    }
    .pageHelper{
  display: flex;
  justify-content: center;
  margin: 40px 0px;
}
  }
</style>

