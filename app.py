import boto3
import sys
import urllib.parse
import logging
import pymysql
import cv2
import os
import numpy as np
import requests
from numpy import save
from numpy import load
from io import BytesIO

s3 = boto3.client('s3')
r3 = boto3.resource('s3')


def handler(event, context):

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    name = os.path.basename(key)

    if key[0:11] != 'descriptors' and key[0:9] != 'keypoints':

        print('Executing descriptor function to: ', name,
              'In bucket: ', bucket, ' path: ', key)
        url = "https://" + bucket + ".s3.sa-east-1.amazonaws.com/" + key

        download(url, dest_folder="/tmp/img/")  # Bajar imagen

        # Obtiene descriptor y guarda el nombre en DB
        descriptor(event, name)

        # Compara los descriptores de la DB con Outfit
        match_descriptors(name, bucket, key)

    else:
        print("Este archivo fue guardado en descriptores o keypoints...")


def descriptor(event, name):

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    img = cv2.imread("/tmp/img/" + name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    name_dk = name[0:-4]
    file_d = name_dk + 'd' + '.npy'
    if not os.path.exists('/tmp/files/'):
        os.makedirs('/tmp/files/')  # create folder if it does not exist
    save('/tmp/files/' + file_d, descriptors)

    keyp = np.asarray(keypoints)
    keyp_str = np.array2string(keyp)
    file_k = name_dk + 'k' + '.npy'
    save('/tmp/files/' + file_k, keyp_str)

    if key[0:6] != 'outfit':
        r3.Bucket(bucket).upload_file(
            '/tmp/files/' + file_d, 'descriptors/' + file_d)
        r3.Bucket(bucket).upload_file(
            '/tmp/files/' + file_k, 'keypoints/' + file_k)
        print("Descriptor and Keypoints OBTAINED and saved to upcyclapp-s3/descriptors")

        mysql_query(event, file_d, file_k)

    delete("/tmp/img/", name)
    delete("/tmp/files/", file_k)


def db_descriptor():

    conn = connection()

    print("Retrieving descriptors and images from DB...")

    descriptor = ([])
    image = ([])

    with conn.cursor() as cur:
        cur.execute(
            'SELECT descriptor, imagen FROM ropas')
        for row in cur:
            if row[0] != None and row[0] != '':
                descriptor.append(row[0])
                image.append(row[1])

    total = np.column_stack((descriptor, image))

    print("Retrieving completed")

    return total


def match_descriptors(name, bucket, key):

    name_dk = name[0:-4]
    file_d = name_dk + 'd' + '.npy'
    f = BytesIO()

    if key[0:6] == 'outfit':
        print('Applying descriptor matching...')
        source = '/tmp/files/'
        descriptor_file = source + file_d
        des = load(descriptor_file)
        res = db_descriptor()
        print('Número de descriptores en DB: ', len(res))
        for i in res:
            print("Comparando ", name_dk, " con ", i[0])
            print('descriptor DB: ')
            obj = s3.download_fileobj('upcyclapp-s3', 'descriptors/' + i[0], f)
            f.seek(0)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, load(f), k=2)
            print("imagen subida: ", key)
            print("imagen db: ", i[1])

            lowe_ratio = 0.89
            good = []

            for m, n in matches:
                if m.distance < lowe_ratio*n.distance:
                    good.append([m])

            msg1 = 'using %s with lowe_ratio %.2f' % (
                "Sift", lowe_ratio)
            msg2 = 'there are %d good matches' % (len(good))

            print(msg1)
            print(msg2)

            save_descriptors(len(good), i[0])  # Guardar score en base de datos

        # Borrar imagen outfit
        delete_S3_Object(bucket, key)

    # Borrar Temporales
    delete("/tmp/files/", file_d)


def save_descriptors(descriptors, name):
    conn = connection()

    with conn.cursor() as desc:
        desc.execute(
            """UPDATE ropas SET dPoints = '""" + str(descriptors) + """' WHERE descriptor = '""" + name + """'""")
        print("Descriptor: " + str(descriptors) + " Saved in: " + name)

    conn.commit()
    conn.close()


def mysql_query(event, descriptors, keypoints):

    conn = connection()

    key = event['Records'][0]['s3']['object']['key']

    with conn.cursor() as cur:
        cur.execute(
            """SELECT ropaId FROM ropas WHERE imagen LIKE '%""" + key + """'""")
        for row in cur:
            print("image id in database:", row[0])

    with conn.cursor() as desc:
        desc.execute(
            """UPDATE ropas SET descriptor = '""" + descriptors + """' WHERE ropaId = '""" + row[0] + """'""")
        print("Descriptor Saved...")
    conn.commit()

    with conn.cursor() as desc:
        desc.execute(
            """UPDATE ropas SET keypoints = '""" + keypoints + """' WHERE ropaId = '""" + row[0] + """'""")
        print("Keypoints Saved...")

    conn.commit()
    conn.close()


def download(url: str, dest_folder: str):

    print("Downloading file...")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    # be careful with file names
    filename = url.split('/')[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("Saving image to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


def delete(location, file):
    data = os.path.join(location, file)
    os.remove(data)
    print(file + " DELETED from docker...")


def delete_S3_Object(bucket, key):
    s3.delete_object(Bucket=bucket, Key=key)
    print("S3 outfit image deleted: " + key)


def connection():
    print("Connecting to DB...")

    rds_host = "upcyclapp-db.ccfycnbxxhbd.sa-east-1.rds.amazonaws.com"
    name = "felipeazs"
    password = "urUf7f*37hf"
    db_name = "upcyclapp"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    try:
        conn = pymysql.connect(host=rds_host, user=name,
                               passwd=password, db=db_name, connect_timeout=5)
    except pymysql.MySQLError as e:
        logger.error(
            "ERROR: Unexpected error: Could not connect to MySQL instance.")
        logger.error(e)
        sys.exit()

    logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")

    return conn
