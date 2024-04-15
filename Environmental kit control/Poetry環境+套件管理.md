# 安裝指南

## 安裝方法一（推薦）

### 步驟 1: 安裝 Scoop

開啟終端機，安裝 Scoop，這是一個專為 Windows 系統設計的命令行安裝器，避免了傳統安裝方式可能造成的 PATH 環境變量污染，它將所有軟體安裝在用戶的 home 目錄下，保持系統的清潔，用來安裝 pipx。

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

[更多關於 Scoop 的資訊](https://scoop.sh/)

更新 Scoop 版本
scoop update

開啟終端機，安裝 pipx ，用於全域安裝 Python CLI 應用程式，同時仍將它們隔離在虛擬環境中。 將用於安裝Poetry時管理升級和卸載。
scoop install pipx
pipx ensurepath
參考：https://pipx.pypa.io/stable/installation/

更新 pipx 版本、查看套件、更新套件
scoop update pipx、pipx list、pipx upgrade-all

安裝方法二 ( 備案 )
官方提供了一個自定義安裝程式，可以在新的虛擬環境中安裝 Poetry 並允許Poetry管理自己的環境，更改安裝路徑請參考下方網址。
Linux: curl -sSL https://install.python-poetry.org | python3 -
windows: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
參考：https://python-poetry.org/docs/#installing-with-the-official-installer

將執行poetry指令執行檔的路經，手動新增至使用者環境變數的PATH (W11)，以下為參考路徑，請依實際個人實際情況更改。
C:\Users\btea4\AppData\Roaming\Python\Scripts

確認是否安裝成功
poetry --version

更新
poetry update

查看配置清單
poetry config --list

設定poetry配置，將虛擬環境安裝在專案中，而不是統一放置虛擬環境
poetry config virtualenvs.in-project true



在專案的根目錄執行，產生pyproject.toml設定檔
poetry init

建立虛擬環境，將會依據剛才設定的pyproject.toml決定環境架構
poetry env use python

啟動，會偵測當前目錄或所屬上層目錄是否存在pyproject.toml來確定所要啟動的虛擬環境
poetry shell

退出
exit



開發環境
安裝套件，這等於pip3 install，而安裝的套件會詳細列在poetry.lock，這等於requirements.txt
poetry add 套件名稱

列出清單，相當於pip3 list
poetry show

列出樹狀圖清單，也是此套件管理的亮點
poetry show --tree

移除套件，當poetry移除時會進行依賴解析，簡單來說可以想像它會根據樹狀圖移除所有相依套件，並且條件是其餘套件都不再依賴它，而pip uninstall只會移除指定套件。
poetry remove 套件名稱


部屬環境
建立部屬環境的套件清單，可以到pyproject.toml清楚看到
poetry add 套件名稱 -D

移出清單
poetry remove 套件名稱 -D



輸出Poetry虛擬環境的requirements.txt，如想去除hash值則使用第二行指令
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --output requirements.txt --without-hashes



移除虛擬環境，不知道為什麼會發生錯誤，直接刪除整個專案
poetry env remove python

移除poetry
curl -sSL https://install.python-poetry.org | python3 - --uninstall



沒有找到能夠安裝PytorchGPU版本的方法，直接pip3不納入poetry管理
https://pytorch.org/get-started/locally/#windows-pip

似乎可行的方案但未經嘗試
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.23.2"
torch = "1.12.1"
torchvision = "0.13.1"

install_cu116:
	poetry install
	poetry run pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
