package com.example.mnist3a

import android.Manifest
import android.graphics.*
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.mnist3a.ml.Degit
import com.innfinity.permissionflow.lib.requestPermissions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    lateinit var txv: TextView
    lateinit var viewFinder: PreviewView
    lateinit var img: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        txv = findViewById(R.id.txv)
        txv.text = "您尚未允許拍照權限"
        viewFinder = findViewById(R.id.viewFinder)
        img = findViewById(R.id.img)

        GlobalScope.launch(Dispatchers.Main) {
            requestPermissions(Manifest.permission.CAMERA)
                .collect { permissions ->
                    txv.text = "用相機辨識手寫數字"
                    startCamera()
                }
        }

    }


    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            //val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA  //自拍

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            var executor = Executors.newSingleThreadExecutor()
            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->

                var bitmap = imageProxy.image?.let { rotateBitmap(toBitmap(it),90f) }
                GlobalScope.launch(Dispatchers.Main) {
                    img.setImageBitmap(bitmap)
                    if (bitmap != null){
                        classifyDrawing(bitmap)
                    }
                }

                // after done, release the ImageProxy object
                imageProxy.close()
            })


            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis)

            } catch(exc: Exception) {
                Toast.makeText(this, "Use case binding failed: ${exc.message}",
                    Toast.LENGTH_SHORT).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }


    fun toBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yBuffer: ByteBuffer = planes[0].buffer
        val uBuffer: ByteBuffer = planes[1].buffer
        val vBuffer: ByteBuffer = planes[2].buffer
        val ySize: Int = yBuffer.remaining()
        val uSize: Int = uBuffer.remaining()
        val vSize: Int = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
    }


    fun classifyDrawing(bitmap : Bitmap) {
        val model = Degit.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28), DataType.FLOAT32)
        var byteBuffer = getByteBufferNormalized(bitmap)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        //比較各數字權重，取最大者
        var max = 0
        var maxValue = outputFeature0.floatArray[0]
        for (i in 1..9){
            if (outputFeature0.floatArray[i] > maxValue){
                max = i
                maxValue = outputFeature0.floatArray[i]
            }
        }
        txv.text = "辨識結果：" + max.toString()

        // Releases model resources if no longer used.
        model.close()
    }

    private fun getByteBufferNormalized(bitmapIn: Bitmap): ByteBuffer {
        val bitmap = Bitmap.createScaledBitmap(bitmapIn, 28, 28, true)
        val width = bitmap.width
        val height = bitmap.height
        val byte_buffer: ByteBuffer = ByteBuffer.allocateDirect(4 * width * height)
        byte_buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (pixel in pixels) {
            byte_buffer.putFloat(Color.blue(pixel).toFloat() / 255.0f)
        }
        return byte_buffer
    }



}