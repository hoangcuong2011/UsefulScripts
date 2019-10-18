I found using os.scandir makes us much faster than os.listdir. This is very important when we work with a folder that has A LOT of files
Here is the piece of code:

    import os
    path = "/home/abcxyz/elastic-data"
    article_ids = os.scandir(path)
    for i, x in enumerate(article_ids):
      if not x.is_file():
        continue
        x = x.name
        if x == '.DS_Store':
          continue
