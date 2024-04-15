# 安裝指南

## 安裝方法一：使用 pipx （推薦）

### 步驟一：安裝 Scoop

這是一個專為 Windows 設計的安裝器，避免了 PATH 環境變量污染，將用於安裝 pipx 時管理升級和卸載。[更多關於 Scoop 的資訊](https://scoop.sh/)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
```powershell
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

### 步驟二：安裝 pipx

用於全域安裝 Python CLI 應用程式，同時仍將它們隔離在虛擬環境中，將用於安裝Poetry時管理升級和卸載。[更多關於 pipx 的資訊](https://pipx.pypa.io/stable/installation/)
```powershell 
scoop install pipx
```
```powershell
pipx ensurepath
```

### 步驟三：安裝 poetry

Poetry 是 Python 中用於依賴管理和打包的工具，它允許聲明專案所依賴的庫，並將為您管理（安裝/更新）它們。[更多關於 poetry 的資訊](https://python-poetry.org/docs/#installing-with-pipx)
```powershell
pipx install pipx
```

## 安裝方法二：使用官方安裝程式 ( 不建議 )

### 步驟一：安裝 poetry
官方提供了一個自定義安裝程式，可以在新的虛擬環境中安裝 Poetry 並允許Poetry管理自己的環境。[更多關於 poetry 的資訊](https://python-poetry.org/docs/#installing-with-the-official-installer)
```windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
```Linux
curl -sSL https://install.python-poetry.org | python3 -
```

## 使用方法

### 開發環境

創建新項目
```powershell
poetry new poetry-demo
```

如果是已存在的專案，執行 portry 初始化建立 pyproject.toml。
```powershell
poetry init
```

該命令從當前專案中讀取 poetry.lock， 解析依賴項並安裝它，如果不存在該檔案則創建。
```powershell
poetry install
```

安裝套件，這等於pip3 install，而安裝的套件會詳細列在poetry.lock，這等於requirements.txt
```powershell
poetry add 套件名稱
```

列出清單，相當於pip3 list
```powershell
poetry show
```

列出樹狀圖清單
```powershell
poetry show --tree
```

移除套件，當poetry移除時會進行依賴解析，簡單來說可以想像它會根據樹狀圖移除所有相依套件，並且條件是其餘套件都不再依賴它，而pip uninstall只會移除指定套件。
```powershell
poetry remove 套件名稱
```

### 部屬環境

建立部屬環境的套件清單，可以到pyproject.toml清楚看到
```powershell
poetry add 套件名稱 -D
```

移出清單，相當於pip3 remove
```powershell
poetry remove 套件名稱 -D
```

輸出Poetry虛擬環境的requirements.txt
```powershell
poetry export -f requirements.txt --output requirements.txt
```
