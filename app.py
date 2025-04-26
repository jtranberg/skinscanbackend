import axios from 'axios';

app.post('/predict', async (req, res) => {
  try {
    const formData = new FormData();
    formData.append('image', req.files.image.data, 'upload.jpg');
    formData.append('age', req.body.age);
    formData.append('gender', req.body.gender);
    formData.append('weight', req.body.weight);
    formData.append('lat', req.body.lat);
    formData.append('lon', req.body.lon);

    const response = await axios.post('http://localhost:5001/predict', formData, {
      headers: formData.getHeaders()
    });

    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Prediction service failed' });
  }
});
