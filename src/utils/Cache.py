
class ImageCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.dict = {}

    def addImage(self, key, image):
        if (len(self.dict) < self.max_size):
            if not key in self.dict:
                self.dict[key] = image.clone()

    def inCache(self, key):
        return key in self.dict
    
    def getImage(self, key):
        if self.inCache(key):
            return self.dict[key].clone()
        else:
            return None
    
    def clearCache(self):
        self.dict = {}

        