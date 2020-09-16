import sys
import os
import requests

import pprint

"""
E.g.:
python SendImage.py ../../../DataSets/ISIC/ISIC-190110/Images/ISIC_0000000.jpeg http://127.0.0.1:5000/classify/binary

"""


if len(sys.argv) < 3:
    print("Send an image through a HTTP POST request to the given URL and reads the answer.")
    print("Usage: SendImage <filename:str> <url:str>")
    print("If the answer is an application/json, it will be printed in the standard output.")
    print("If the answer is an image/'extension', it will be save as '<filename>-answer.<extension>'.")
    exit(10)

image_filepath = sys.argv[1]
destination_url = sys.argv[2]

print("Sending image {} to url {}".format(image_filepath, destination_url))

if not os.path.exists(image_filepath):
    raise Exception("Path {} doesn't exist".format(image_filepath))

if not os.path.isfile(image_filepath):
    raise Exception("Path {} is not a file".format(image_filepath))

_, filename = os.path.split(image_filepath)
filename_root, filename_extension = os.path.splitext(filename)
content_type = 'image/' + filename_extension
files = {'file': (filename, open(image_filepath, 'rb'), content_type)}

response = requests.post(url=destination_url, files=files)

print("Answer code: {}".format(response.status_code))

if response.status_code == requests.codes.ok:

    print("Got response status OK")
    response_content_type = response.headers["Content-Type"]
    print(response_content_type)

    #
    # For images
    if response_content_type.startswith("image/"):
        answer_img_extension = response_content_type[len("image/"):]
        save_image_filename = filename_root + "-answer." + answer_img_extension
        print("Saving to " + save_image_filename)
        with open(save_image_filename, 'wb') as f:
            f.write(response.content)
        pass

    #
    # For JSON content
    elif response_content_type.startswith("application/json"):
        pprint.pprint(response.json())

    #
    # For plain HTML content
    elif response_content_type.startswith("text/html"):
        print(response.content)

    #
    # If we don't know what to do with it
    else:
        print("Unrecognized content type. Passing.")

else:
    print("There was an HTTP error: {}".format(response.reason))


print("Done.")
