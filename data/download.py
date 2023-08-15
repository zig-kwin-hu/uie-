import gdown

url = "https://drive.google.com/uc?id=1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT"
output = "IE_INSTRUCTION.zip"
gdown.download(url, output, quiet=False)