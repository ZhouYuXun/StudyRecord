# SeaweedFS Guide

## Table of Contents
- [SeaweedFS Guide](#seaweedfs-guide)
  - [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Purpose](#purpose)
  - [Scope](#scope)
- [Install](#install)
  - [Server](#server)
  - [Python SDK (Boto3)](#python-sdk-boto3)
- [Usage](#usage)
  - [Connection](#connection)
  - [Create Bucket](#create-bucket)
  - [Upload Object](#upload-object)
  - [Query Object](#query-object)
  - [Delete Object](#delete-object)
  - [Download Object](#download-object)
  - [Delete Bucket](#delete-bucket)

# Introduction

## Purpose

本指南旨在介紹如何結合使用 SeaweedFS 的 S3 介面和 Boto3，實作一個簡單的物件儲存系統。我們將詳細介紹建立儲存桶、上傳和下載對象，以及如何刪除對象和儲存桶的過程。

## Scope

SeaweedFS 是一種多功能且高效的儲存系統，旨在滿足現代系統管理員管理 blob、物件、檔案和資料倉儲需求的混合需求。

無論資料集有多大，其架構都可以透過恆定時間 (O(1)) 磁碟查找來確保快速存取時間。 這使其成為速度和效率至關重要的環境的絕佳選擇。

為了實現這些目標，SeaweedFS 引入了幾個核心概念，包括節點（Node）、資料中心（DataCenter）、機架（Rack）、儲存節點（DataNode）、邏輯卷（Volume）、針（Needle）和檔案集（Collection）。

這些概念彼此相互作用，構成了 SeaweedFS 的基礎架構，下面我們將對它們進行詳細介紹：

1. 節點 (Node): 在 SeaweedFS 中，節點是系統架構的基本單位，它可以是一台服務器或是服務器集群的一部分。節點被進一步分類為資料中心、機架和儲存節點，以模擬真實世界中數據中心的物理結構。

2. 資料中心 (DataCenter): 資料中心是一組相關節點的集合，代表實際世界中的一個或多個數據中心。在 SeaweedFS 中，資料中心用於組織數據儲存的地理分布，有助於實現數據的地理冗餘和高可用性。

3. 機架 (Rack): 機架是資料中心內部的一個子集，代表一組在物理位置或網絡拓撲上相近的節點。機架的概念使得 SeaweedFS 能夠更細致地管理數據的冗餘和負載均衡。

4. 儲存節點 (DataNode): 儲存節點是存儲和管理數據的物理或虛擬機器。每個儲存節點負責一組邏輯卷，並處理對這些卷中數據的讀寫請求。

5. 邏輯卷 (Volume): 邏輯卷是儲存數據的容器，它可以分佈在多個儲存節點上。每個邏輯卷包含多個針，這些針對應於儲存的檔案。邏輯卷的設計允許 SeaweedFS 高效地管理和訪問大量的小檔案。

6. 針 (Needle): 針是邏輯卷中的基本數據單位，代表一個存儲在卷中的檔案。針的概念使得檔案的存取非常高效，特別是在處理大量小檔案時。

7. 檔案集 (Collection): 檔案集是一組邏輯相關的檔案，這些檔案可以分佈在多個邏輯卷上。檔案集的概念使得用戶能夠更方便地組織和管理相關檔案。

通過這些概念的整合，SeaweedFS 提供了一個高度靈活和可擴展的分佈式檔案儲存系統，適用於從小型應用到大規模企業級部署的各種場景。

SeaweedFS 支持多種儲存後端，包括本地硬碟、S3 兼容儲存和 HDFS，讓用戶可以根據自己的需求選擇最合適的儲存解決方案。它還支持自動數據複製和故障轉移，確保數據的高可用性和持久性。

SeaweedFS 的一個關鍵特點是其高性能。通過有效的數據索引和快速的數據讀寫能力，SeaweedFS 能夠支援大量的同時讀寫操作，使其成為適合大數據應用和即時數據處理場景的理想選擇。

SeaweedFS 是開源項目，遵循 Apache 2.0 許可證。這意味著任何人都可以自由地使用、修改和分發它，無論是用於商業還是非商業目的。SeaweedFS 的源代碼托管在 GitHub 上，擁有一個活躍的開發者社群，不斷地貢獻新功能和改進。

總結來說，SeaweedFS 提供了一個靈活、高效和可靠的分散式檔案儲存解決方案，非常適合處理大規模數據儲存和高性能數據處理

更多詳細資料請參閱官方文件：https://github.com/seaweedfs/seaweedfs/wiki

# Install

## Server

請先決定 SeaweedFS 根目錄後，開啟該位置終端機執行以下指令，將創建資料夾並下載設定檔

windows

```bash
New-Item -ItemType Directory -Force -Path docker/prometheus; Invoke-WebRequest -Uri https://raw.githubusercontent.com/seaweedfs/seaweedfs/master/docker/seaweedfs-compose.yml -OutFile docker/seaweedfs-compose.yml; Invoke-WebRequest -Uri https://raw.githubusercontent.com/seaweedfs/seaweedfs/master/docker/prometheus/prometheus.yml -OutFile docker/prometheus/prometheus.yml
```

linux

```bash
mkdir -p docker/prometheus && curl -L https://raw.githubusercontent.com/seaweedfs/seaweedfs/master/docker/seaweedfs-compose.yml -o docker/seaweedfs-compose.yml && curl -L https://raw.githubusercontent.com/seaweedfs/seaweedfs/master/docker
```

確認是否安裝 Docker，在終端機中輸入 docker --version。如果未安裝或找不到 Docker 指令，請訪問官方網站下載並安裝。下載並安裝請參閱官方文件：https://www.docker.com/get-started/

如果是 linux 環境，則需另外安裝 docker-compose，下載並安裝請參閱官方文件：https://docs.docker.com/compose/install/standalone/，安裝完畢後使用 docker-compose 啟動 SeaweedFS 伺服器

```bash
docker-compose -f docker/seaweedfs-compose.yml -p seaweedfs up
```

恭喜，現在可以嘗試開啟架設完畢的 SeaweedFS 伺服器，點選以下連結確認伺服器是否正常運作。

SeaweedFS 所使用的 Port：
- Master 節點負責整個 SeaweedFS 集群的管理和協調。它維護著集群的元數據，包括儲存卷的位置和狀態信息。SeaweedFS Master：http://localhost:9333/
- Filer 提供了一個類似於傳統檔案系統的介面，允許用戶通過檔案路徑來儲存和訪問數據，而不僅僅是通過 Volume ID。SeaweedFS Filer：http://localhost:8888/
- Prometheus 監控被整合進 SeaweedFS，允許用戶監控和收集關於集群性能的指標數據，如請求率、錯誤率、處理延時等。SeaweedFS prometheus：http://localhost:9000/

更多詳細資料請參閱官方文件：https://github.com/seaweedfs/seaweedfs/wiki/Production-Setup 與 https://prometheus.io/docs/introduction/overview/

當您想要停止伺服器的運行時，在終端機處按壓 CTRL+C，Docker Compose 會嘗試優雅地停止正在運行的容器，然後執行 docker-compose -f docker/seaweedfs-compose.yml down 完成關閉，重新運行只需要再次輸入啟動指令即可。

## Python SDK (Boto3)

Boto3 是 Amazon Web Services (AWS)的官方 Python 軟件開發套件（SDK），提供了一種在 Python 應用程序中訪問和管理AWS服務的方式。透過 Boto3，開發者可以使用Python代碼輕鬆地與AWS的各種服務進行交互，如 Amazon S3（簡單存儲服務）、Amazon EC2（彈性計算雲）、Amazon DynamoDB（NoSQL數據庫服務）等，涵蓋了 AWS 提供的絕大多數服務。

而該 SDK 由兩個關鍵的 Python 包組成：Botocore（提供低級的庫 Python 開發工具包和 AWS CLI 之間共用的功能）和Boto3（實現 Python SDK 本身），而 SeaweedFS 支援 S3 接口，在這次的範例中，我們將使用 Boto3 的 S3 接口來實現與 SeaweedFS 的連接，並進行一些基本的操作。

```bash
pip3 install boto3
```

更多詳細資料請參閱官方文件：https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html

# Usage

在此介紹簡易的基礎指令，並提供 Jupyter 的直觀執行範例。

## Connection

這一步驟是確立與 SeaweedFS S3 接口的連接，對於啟動所有後續的操作至關重要，確保您的應用程序可以無縫地與雲存儲系統集成。

```python
import boto3
# from botocore.client import Config

# 設定連接SeaweedFS的S3接口
s3 = boto3.resource(
    service_name='s3',
    endpoint_url='http://localhost:8333',
    aws_access_key_id='None',
    aws_secret_access_key='None',
    # config=Config()
)
```
更多設定請參閱 Config 物件

## Create Bucket

桶是存儲物件的基本容器，創建桶是組織雲端數據的首步，有助於高效的數據管理和檢索。

```python
bucket_00 = s3.Bucket('0216-00')
```

```python
bucket_00.create()
```

```python
for bucket in s3.buckets.all():
    print(bucket)
```

## Upload Object

Filename 為上傳的資料路徑，Key 為虛擬的資料儲存路徑，上傳成功後可至 http://localhost:8888/ 查看變化。

```python
bucket_00.upload_file(Filename='科別分析_Big5.csv', Key='測試/學校/附中/科別分析_Big5.csv', ExtraArgs={'Metadata': {'name': 'XXXXX', 'number': '0922444666'}})
```

```python
for obj in bucket_00.objects.all():
    print(obj)
```

## Query Object

前缀查詢，查詢帶有該前缀的資料，前缀連續且完整無法省略與跳過。

```python
for obj in bucket_00.objects.filter(Prefix='測試/學校'):
    print(obj)
```

得知 Key 後，使用鍵查詢，獲得完整元數據資料，須知道完整的自定義 Key 值。

```python
s3.meta.client.head_object(Bucket='0216-00', Key='測試/學校/附中/科別分析_Big5.csv')
```

## Delete Object

刪除指定桶特定前缀的物件，此方法適用於當你想要刪除具有特定前缀的所有物件時。

```python
for obj in bucket_00.objects.filter(Prefix=''):
    print(obj)
    obj.delete()
```

精確指定要刪除的物件的 Key。此方法適用於當你知道確切的物件 Key 並且只想刪除該特定物件時。

```python
s3.Object(bucket_name='0216-00', key='測試/學校/附中/科別分析_Big5.csv').delete()
```

## Download Object

Key 參數指定了 S3 桶中物件的路徑和名稱，而 Filename 參數則定義了物件下載後存儲在本地系統上的完整路徑和物件名。

```python
bucket_00.download_file(Key='測試/學校/附中/科別分析_Big5.csv', Filename='D:/科別分析_Big5.csv')
```

## Delete Bucket

它會遍歷每個桶，輸出桶的名稱以確認，然後進行刪除，清除存儲空間並有效管理成本的關鍵維護操作，是數據生命周期管理中的最後一步。

```python
for bucket in s3.buckets.all():
    print(bucket)
    bucket.delete()
```
