from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

    def clean_image(self):
        image = self.cleaned_data.get('image')
        # Check if the image type is valid
        if image.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise forms.ValidationError("Only JPEG and PNG images are allowed.")
        return image
