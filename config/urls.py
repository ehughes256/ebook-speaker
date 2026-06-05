from django.urls import include, path

urlpatterns = [
    path("", include("reader.urls")),
]
