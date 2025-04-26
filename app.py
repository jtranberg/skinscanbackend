import axios from 'axios';
import FormData from 'form-data'; // ✅ Important import

app.post('/predict', async (req, res) => {
  try {
    if (!req.files || !req.files.image) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const formData = new FormData();
    formData.append('image', req.files.image.data, 'upload.jpg');
    formData.append('age', req.body.age);
    formData.append('gender', req.body.gender);
    formData.append('weight', req.body.weight);
    formData.append('lat', req.body.lat);
    formData.append('lon', req.body.lon);

    const response = await axios.post(
      'https://skinscanbackend.onrender.com/predict',
      formData,
      { headers: formData.getHeaders() }
    );

    res.json(response.data);
  } catch (err) {
    console.error('❌ Prediction proxy error:', err.message || err);
    res.status(500).json({ error: 'Prediction service failed' });
  }
});
