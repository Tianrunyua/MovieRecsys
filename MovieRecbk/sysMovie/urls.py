from django.urls import re_path as url
from django.urls import path
from . import views

urlpatterns = [
    path('login', views.login_func),

    path('quit', views.quit_func),

    url('rankingList', views.rankingList_func),
    url('HomeList', views.HomeList_func),

    url('getAllTag', views.getAllTag_func),
    url('findtag', views.findtag_func),

    url('getMovieInfo', views.getMovieInfo_func),
    url('watchrecord', views.watchrecord_func),
    url('getTsimMovies', views.getTsimMovies_func),
    url('getVsimMovies', views.getVsimMovies_func),
    url('getcomMovies', views.getcomMovies_func),

    url('find', views.find_func),

    url('getUserInfo', views.getUserInfo_func),
    url('editUser', views.editUser_func),

    url('RS', views.getRecommend_func),
]