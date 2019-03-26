import json
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import numpy as np
import argparse
import cv2
import time
from PIL import Image
import pytesseract
import os
# from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt


@csrf_exempt
def monobrand(request):
    responseData = {
        'id': 4,
        'name': 'Test Response',
        'roles': ['Admin', 'User']
    }
    if request.method == 'POST' and request.FILES['img']:
        print('HAHAHAHAHAHAHAAHAHAHAHAH')
        file = request.FILES['img']
        print(file.name)           # Gives name
        file.content_type   # Gives Content type text/html etc
        file.size           # Gives file's size in byte
        file.read()         # Reads file
        fs = FileSystemStorage()
        filename = fs.save('./images/'+file.name, file)

    return JsonResponse(responseData)
