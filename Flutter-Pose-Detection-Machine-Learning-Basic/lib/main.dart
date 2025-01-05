import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage(
        title: 'screen',
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  dynamic controller;
  bool isBusy = false;
  bool isRecording = false;
  late Size size;
  late PoseDetector poseDetector;
  late var interpreter;
  List<Pose>? _scanResults;
  CameraImage? img;
  List<Map<PoseLandmarkType, PoseLandmark>> recordedPoses = [];

  final List<PoseLandmarkType> landmarkOrder = [
    PoseLandmarkType.rightHeel,
    PoseLandmarkType.rightHip,
    PoseLandmarkType.rightIndex,
    PoseLandmarkType.rightKnee,
    PoseLandmarkType.rightPinky,
    PoseLandmarkType.rightShoulder,
    PoseLandmarkType.rightThumb,
    PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftWrist,
    PoseLandmarkType.leftMouth,
    PoseLandmarkType.rightMouth,
    PoseLandmarkType.nose,
    PoseLandmarkType.rightAnkle,
    PoseLandmarkType.rightEar,
    PoseLandmarkType.rightElbow,
    PoseLandmarkType.rightEye,
    PoseLandmarkType.rightEyeInner,
    PoseLandmarkType.rightEyeOuter,
    PoseLandmarkType.rightFootIndex,
    PoseLandmarkType.leftAnkle,
    PoseLandmarkType.leftEar,
    PoseLandmarkType.leftElbow,
    PoseLandmarkType.leftEye,
    PoseLandmarkType.leftEyeInner,
    PoseLandmarkType.leftEyeOuter,
    PoseLandmarkType.leftFootIndex,
    PoseLandmarkType.leftHeel,
    PoseLandmarkType.leftHip,
    PoseLandmarkType.leftIndex,
    PoseLandmarkType.leftKnee,
    PoseLandmarkType.leftPinky,
    PoseLandmarkType.leftShoulder,
    PoseLandmarkType.leftThumb,
  ];

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

  initializeCamera() async {
    final options = PoseDetectorOptions(mode: PoseDetectionMode.stream);
    poseDetector = PoseDetector(options: options);

    controller = CameraController(cameras[0], ResolutionPreset.high);
    await controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      controller.startImageStream((image) {
        if (!isBusy && isRecording) {
          isBusy = true;
          img = image;
          doPoseEstimationOnFrame();
        }
      });
    });
  }

  Map<PoseLandmarkType, PoseLandmark> normalizeLandmarks(Map<PoseLandmarkType, PoseLandmark> landmarks, Size imageSize) {
    return landmarks.map((type, landmark) {
      return MapEntry(type, PoseLandmark(
        type: landmark.type,
        x: landmark.x / imageSize.width,
        y: landmark.y / imageSize.height,
        z: landmark.z, // Z-coordinate might not need normalization
        likelihood: landmark.likelihood,
      ));
    });
  }

  doPoseEstimationOnFrame() async {
    try {
      final inputImage = getInputImage();
      _scanResults = await poseDetector.processImage(inputImage);

      if (isRecording) {
        final Size imageSize = Size(img!.width.toDouble(), img!.height.toDouble());
        _scanResults!.forEach((pose) {
          recordedPoses.add(normalizeLandmarks(pose.landmarks, imageSize));
        });
      }

      setState(() {
        isBusy = false;
        _scanResults;
      });
    } catch (e) {
      setState(() {
        isBusy = false;
      });
    }
  }

  InputImage getInputImage() {
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in img!.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();
    final Size imageSize = Size(img!.width.toDouble(), img!.height.toDouble());
    final camera = cameras[0];
    final imageRotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    // if (imageRotation == null) return;

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(img!.format.raw);
    // if (inputImageFormat == null) return null;

    final planeData = img!.planes.map(
      (Plane plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList();

    final inputImageData = InputImageData(
      size: imageSize,
      imageRotation: imageRotation!,
      inputImageFormat: inputImageFormat!,
      planeData: planeData,
    );
    final inputImage =
        InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);

    return inputImage;
  }

  Widget buildResult() {
    if (_scanResults == null ||
        controller == null ||
        !controller.value.isInitialized) {
      return const Text('Empty');
    }

    final Size imageSize = Size(
      controller.value.previewSize!.height,
      controller.value.previewSize!.width,
    );

    CustomPainter painter = PosePainter(imageSize, _scanResults!);

    return CustomPaint(
      painter: painter,
    );
  }

  void startRecording() {
    setState(() {
      isRecording = true;
      recordedPoses.clear();
    });
  }

  void stopRecording() {
    setState(() {
      isRecording = false;
      controller.stopImageStream();
      compileAndSendYCoordinates(); // Call the method when recording stops
    });
  }

  void compileAndSendYCoordinates() async {
    List<List<double>> yCoordinates = [];

    for (var pose in recordedPoses) {
      List<double> yCoords = [];
      for (var landmark in landmarkOrder) {
        yCoords.add(pose[landmark]?.y ?? 0.0);
      }
      yCoordinates.add(yCoords);
    }

    // Ensure each list has 33 elements
    if (yCoordinates.length < 33) {
      while (yCoordinates.length < 33) {
        yCoordinates.add(List.filled(847, 0.0));
      }
    } else if (yCoordinates.length > 33) {
      yCoordinates = yCoordinates.sublist(0, 33);
    }

    // Pad or trim the coordinates to ensure each list has 847 elements
    for (var i = 0; i < yCoordinates.length; i++) {
      if (yCoordinates[i].length < 847) {
        while (yCoordinates[i].length < 847) {
          yCoordinates[i].add(0.0);
        }
      } else if (yCoordinates[i].length > 847) {
        yCoordinates[i] = yCoordinates[i].sublist(0, 847);
      }
    }

    // Ensure the shape is [1, 33, 847]
    List<List<List<double>>> array = [yCoordinates];

    // Print the shape of the yCoordinates
    print('Shape of yCoordinates: [${array.length}, ${array[0].length}, ${array[0][0].length}]');

    var url = Uri.parse('https://testinghaha.onrender.com/endpoint');
    try {
      var response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(array),
      );

      if (response.statusCode == 200) {
        var output = jsonDecode(response.body);
        print('Model output: $output');
        print('Flask is connected');
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => OutputPage(output: output),
          ),
        );
      } else {
        print('Failed to get model output');
        print('Flask is not connected');
        print('Response status: ${response.statusCode}');
        print('Response body: ${response.body}');
      }
    } catch (e) {
      print('Error occurred: $e');
    }
  }

  Widget buildRecordingButton() {
    return Positioned(
      bottom: 20.0,
      left: size.width / 2 - 50,
      child: ElevatedButton(
        onPressed: isRecording ? stopRecording : startRecording,
        child: Icon(
          isRecording ? FontAwesomeIcons.stop : FontAwesomeIcons.video,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget buildRecordedPoses() {
    if (recordedPoses.isEmpty || isRecording) return Container();
    controller.stopImageStream();
    return Positioned(
      top: 0.0,
      left: 0.0,
      width: size.width,
      height: size.height,
      child: Container(
        color: Colors.black.withOpacity(0.7),
        child: SingleChildScrollView(
          child: Center(
            child: Column(
              children: [
                Text(
                  recordedPoses.map((pose) {
                    return pose.entries.map((entry) {
                      return '${entry.key}: (${entry.value.x.toStringAsFixed(2)}, ${entry.value.y.toStringAsFixed(2)})';
                    }).join('\n');
                  }).join('\n\n'),
                  style: const TextStyle(color: Colors.white),
                ),
                const SizedBox(height: 10),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    List<Widget> stackChildren = [];
    size = MediaQuery.of(context).size;
    if (controller != null) {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height,
          child: Container(
            child: (controller.value.isInitialized)
                ? AspectRatio(
                    aspectRatio: controller.value.aspectRatio,
                    child: CameraPreview(controller),
                  )
                : Container(),
          ),
        ),
      );

      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height,
          child: buildResult(),
        ),
      );

      stackChildren.add(buildRecordingButton());
      stackChildren.add(buildRecordedPoses());
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Pose Detection",
          style: TextStyle(color: Colors.black),
        ),
        backgroundColor: Colors.blue,
        actions: [
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () {
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (context) => MyHomePage(title: 'screen')),
                (Route<dynamic> route) => false,
              );
            },
          ),
        ],
      ),
      backgroundColor: Colors.black,
      body: Container(
        margin: const EdgeInsets.only(top: 0),
        color: Colors.black,
        child: Stack(
          children: stackChildren,
        ),
      ),
    );
  }
}

class PosePainter extends CustomPainter {
  PosePainter(this.absoluteImageSize, this.poses);

  final Size absoluteImageSize;
  final List<Pose> poses;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..color = Colors.green;

    final leftPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.yellow;

    final rightPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.blueAccent;

    for (final pose in poses) {
      pose.landmarks.forEach((_, landmark) {
        canvas.drawCircle(
            Offset(landmark.x * scaleX, landmark.y * scaleY), 1, paint);
      });

      void paintLine(
          PoseLandmarkType type1, PoseLandmarkType type2, Paint paintType) {
        final PoseLandmark joint1 = pose.landmarks[type1]!;
        final PoseLandmark joint2 = pose.landmarks[type2]!;
        canvas.drawLine(Offset(joint1.x * scaleX, joint1.y * scaleY),
            Offset(joint2.x * scaleX, joint2.y * scaleY), paintType);
      }

      //Draw arms
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, leftPaint);
      paintLine(
          PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow,
          rightPaint);
      paintLine(
          PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, rightPaint);

      //Draw Body
      paintLine(
          PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip,
          rightPaint);

      //Draw legs
      paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, leftPaint);
      paintLine(
          PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, leftPaint);
      paintLine(
          PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, rightPaint);
      paintLine(
          PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, rightPaint);
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) {
    return oldDelegate.absoluteImageSize != absoluteImageSize ||
        oldDelegate.poses != poses;
  }
}

class OutputPage extends StatelessWidget {
  final dynamic output;

  const OutputPage({Key? key, required this.output}) : super(key: key);

  String getClassLabel(List<dynamic> output) {
    if (output.isEmpty || output[0].isEmpty) return 'Unknown class';
    List<double> values = List<double>.from(output[0]);
    double maxValue = values.reduce((a, b) => a > b ? a : b);
    int maxIndex = values.indexOf(maxValue);

    switch (maxIndex) {
      case 0:
        return "The class is normal";
      case 1:
        return "The class is Parkinson's mild severity";
      case 2:
        return "The class is Parkinson's high severity";
      default:
        return "Unknown class";
    }
  }

  @override
  Widget build(BuildContext context) {
    String classLabel = getClassLabel(output);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Output'),
        actions: [
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () {
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (context) => MyHomePage(title: 'screen')),
                (Route<dynamic> route) => false,
              );
            },
          ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Model output: $output'),
            const SizedBox(height: 10),
            Text(classLabel),
          ],
        ),
      ),
    );
  }
}
