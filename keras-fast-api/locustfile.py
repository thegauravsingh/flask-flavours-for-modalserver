from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(0.5, 3.0)
    def on_start(self):
        """ on_start is called when the TaskSet is starting """
        pass
    def on_stop(self):
        """ on_stop is called when the TaskSet is stopping """
        pass        

    @task(1)
    def predict(self):
        with open('dog.jpg', 'rb') as image:
            self.client.post('/predict', files={'img_file': image})
