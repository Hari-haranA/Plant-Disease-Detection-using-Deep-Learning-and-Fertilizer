from django.db import models

class CropDetails(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name

class FertilizerRecommendation(models.Model):
    crop = models.ForeignKey(CropDetails, on_delete=models.CASCADE)
    recommendation = models.TextField()

    def __str__(self):
        return f"Recommendation for {self.crop.name}"
