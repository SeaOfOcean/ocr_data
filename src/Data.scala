import java.io.File

import com.intel.analytics.OpenCV
import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.HashMap

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Data {
  OpenCV.load()

  def main(args: Array[String]): Unit = {
    val bg = Imgcodecs.imread("data/bg/bg.jpg")
    bg.convertTo(bg, CvType.CV_32FC3)
    val bgPixels = getPixels(bg)
    var wStart = 170
    val hStart = 180
    val money = Array[String]("data/num_cn/1.png",
      "data/unit/10000.png",
      "data/num_cn/3.png",
      "data/unit/1000.png",
      "data/num_cn/5.png",
      "data/unit/100.png",
      "data/num_cn/0.png",
      "data/num_cn/7.png",
      "data/unit/1.png",
      "data/unit/z.png")
    money.foreach(f => {
      val data = Imgcodecs.imread(f, -1)
      val number = augment(data)
      val numberPixels = getPixels(number)
      overlay(number, bg, numberPixels, bgPixels, wStart, hStart)
      val interval = Random.nextInt(5) + 30
      wStart += interval
    })


    bg.put(0, 0, bgPixels)
    Imgcodecs.imwrite("data/result/tmp.jpg", bg)
  }

  def overlay(number: Mat, bg: Mat,
    numberPixels: Array[Float], bgPixels: Array[Float],
    wStart: Int, hStart: Int): Unit = {
    var i = 0
    (0 until number.height()).foreach(h => {
      (0 until number.width()).foreach(w => {
        // not transparent
        if (numberPixels(i + 3) > 0) {
          setValue(bgPixels, bg.width(), bg.height(), bg.channels(), w + wStart, h + hStart, 0, 0)
          setValue(bgPixels, bg.width(), bg.height(), bg.channels(), w + wStart, h + hStart, 1, 0)
          setValue(bgPixels, bg.width(), bg.height(), bg.channels(), w + wStart, h + hStart, 2, 0)
        }
        i += 4
      })
    })
  }

  def augment(mat: Mat): Mat = {
    var data = mat
    if (Random.nextDouble() > 0.7) {
      val rotateAngle = (Random.nextDouble() * 10) * (Math.pow(-1, Random.nextInt(2)))
      val data = rotateImage1(mat, rotateAngle)
    }
    val size = Random.nextInt(10) + 60
    Imgproc.resize(data, data, new Size(size, size), 0, 0, 1)
    data
  }

  import org.opencv.core.CvType
  import org.opencv.core.Mat
  import org.opencv.core.Scalar
  import org.opencv.imgproc.Imgproc

  def rotateImage1(img: Mat, degree: Double): Mat = {
    val angle = degree * Math.PI / 180.0
    val a = Math.sin(angle)
    val b = Math.cos(angle)
    val width = img.width
    val height = img.height
    val width_rotate = (height * Math.abs(a) + width * Math.abs(b)).toInt
    val height_rotate = (width * Math.abs(a) + height * Math.abs(b)).toInt
    var map_matrix = new Mat(2, 3, CvType.CV_32F)
    val center = new Point(width / 2, height / 2)
    map_matrix = Imgproc.getRotationMatrix2D(center, degree, 1.0)
    map_matrix.put(0, 2, map_matrix.get(0, 2)(0) + (width_rotate - width) / 2)
    map_matrix.put(1, 2, map_matrix.get(1, 2)(0) + (height_rotate - height) / 2)
    val rotated = new Mat
    Imgproc.warpAffine(img, rotated, map_matrix, new Size(width_rotate, height_rotate))
    rotated
  }

  def getPixels(input: Mat): Array[Float] = {
    val floats = new Array[Float](input.height() * input.width() * input.channels())
    if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(input, CvType.CV_32FC3)
    }
    require(floats.length >= input.channels() * input.height() * input.width())
    input.get(0, 0, floats)
    floats
  }

  def setValue(data: Array[Float], width: Int, height: Int, channel: Int, x: Int, y: Int, c: Int, value: Float): Unit = {
    data(channel * width * y + channel * x + c) = value
  }
}