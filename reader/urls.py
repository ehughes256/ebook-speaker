from django.urls import path
from reader import views

app_name = "reader"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("process/", views.process_view, name="process"),
    path("progress/<str:content_hash>/", views.progress_view, name="progress"),
    path("stream/<str:content_hash>/", views.stream_view, name="stream"),
    path("results/<str:content_hash>/", views.results_view, name="results"),
    path("delete/<str:content_hash>/", views.delete_view, name="delete"),
    path("voice/<str:content_hash>/<str:slug>/", views.voice_view, name="voice"),
    path("voice/<str:content_hash>/<str:slug>/regenerate/", views.regenerate_voice_view, name="regenerate_voice"),
    path("speaker/<str:content_hash>/<str:slug>/update/", views.update_speaker_view, name="update_speaker"),
    path("audio/<str:content_hash>/", views.full_audio_view, name="full_audio"),
    path("compile-all/<str:content_hash>/", views.compile_all_view, name="compile_all"),
    path("compile/<str:content_hash>/", views.compile_view, name="compile"),
    path("compile/<str:content_hash>/stream/", views.compile_stream_view, name="compile_stream"),
    path("results/<str:content_hash>/chapter/<int:chapter>/", views.chapter_content_view, name="chapter_content"),
    path("compile/<str:content_hash>/<int:chapter>/", views.compile_view, name="compile_chapter"),
    path("compile/<str:content_hash>/<int:chapter>/stream/", views.compile_stream_view, name="compile_stream_chapter"),
    path("audio/<str:content_hash>/<int:chapter>/", views.full_audio_view, name="full_audio_chapter"),
    path("cover/<str:content_hash>/", views.cover_view, name="cover"),
    path("listen/", views.listen_list_view, name="listen"),
    path("listen/<str:content_hash>/", views.listen_book_view, name="listen_book"),
]
