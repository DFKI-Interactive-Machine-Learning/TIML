var myApp = angular.module('myApp', ['rzSlider']);

myApp.config(function ($httpProvider){
       	  //$httpProvider.defaults.transformRequest.unshift($httpParamSerializerJQLikeProvider.$get());
       	  $httpProvider.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8';
       	});


myApp.directive('ngFileModel', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function (scope, element, attrs) {
         console.log("Attrs ", attrs)
            var model = $parse(attrs.ngFileModel);
            var isMultiple = attrs.multiple;
            var modelSetter = model.assign;
            element.bind('change', function () {
                var values = [];
                angular.forEach(element[0].files, function (item) {
                    var value = {
                       // File Name 
                        name: item.name,
                        //File Size 
                        size: item.size,
                        //File URL to view 
                        url: URL.createObjectURL(item),
                        // File Input Value 
                        _file: item
                    };
                    values.push(value);
                });
                scope.$apply(function () {
                    if (isMultiple) {
                        modelSetter(scope, values);
                    } else {
                        modelSetter(scope, values[0]);
                    }
                });
            });
        }
    };
}]);

myApp.directive('keyispressed', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function(scope, element, attrs){
            var model = $parse(attrs.ngModel);
            var modelSetter = model.assign;
            element.on('keydown', function (e) {
                if (!scope.keyIsDown && e.key=='Shift') {
                    console.log(" Shift Key down... e ", attrs);
                    modelSetter(scope, true);
                    }
                    });
             element.on('keyup', function (e) {
             if (scope.keyIsDown) {
                console.log(" Key up... e ", e);
                scope.keyIsDown = false;
                }
                });
                    }
                }
}])

myApp.directive('upload', ['$parse', function($parse) {
    return {
      restrict: 'EA',
      replace: true,
      //scope: {},
      //require: '?ngModel',
      //template: '<div class="asset-upload">Drop files here</div>',
      link: function(scope, element, attrs, ngModel) {
        var model = $parse(attrs.ngFileModel);
        var isMultiple = attrs.multiple=="true";
        var modelSetter = model.assign;
        element.on('dragover', function(e) {
            //console.log("dragover  ", e);
            e.preventDefault();
            e.stopPropagation();
        });
        element.on('dragenter', function(e) {
            //console.log("dragenter target ", dragTarget, "ev ",e);
            e.preventDefault();
            e.stopPropagation();
            dragTarget = e.target;
            e.dataTransfer.setData("Text",e.target.id);
        });
        element.on('drop', function(e) {
            console.log("Drop Event files " , e.dataTransfer, e.dataTransfer.files);
            console.log("FileList: ",e.dataTransfer.files[0])
            //console.log("original " , e.originalEvent);
            e.preventDefault();
            e.stopPropagation();
            var values = [];
            if (e.dataTransfer){
                if (e.dataTransfer.files.length > 0) {
                    angular.forEach(e.dataTransfer.files, function (item) {
                     var value = {
                       // File Name
                        name: item.name,
                        //File Size
                        size: item.size,
                        //File URL to view
                        url: URL.createObjectURL(item),
                        // File Input Value
                        _file: item
                    };
                    values.push(value);
                    })
                    scope.$apply(function () {
                    console.log("Dropped Files: " , values, isMultiple);
                    if (isMultiple) {
                         modelSetter(scope, values);
                    } else {
                        modelSetter(scope, values[0]);
                    }
                })
                }
                } else { console.log("no data transfer in event", e);
                }

            return false;
        });

        }
    };
}]);



myApp.controller('myCtrl', ['$scope', '$http', '$sce', '$window', /*'fileUpload',*/


function($scope, $http, $sce, $window, fileUpload)
{

    $scope.uploadStatus = "";
    $scope.uploadedFile = {};
    $scope.classifiedImages = {};
	$scope.classificationResults = [];
	$scope.imgIdx = 0;
    $scope.abcdResults = [];
    $scope.loadingGif = "loadingwedges.gif";
    $scope.hoverOn = true;

    $scope.testWithSamples = false;
    $scope.classifyTestImages = $window.classifyTestImages;

    var files = [];

    var vm = this;

    var currentDraggedImage = null;
    $scope.imgOnDragStart = function (event) {
    	currentDraggedImage = event.srcElement.currentSrc;
    	console.log("img event start", event.srcElement.currentSrc);
    	
    }
    $scope.opaSlider = {
        value: 80,
        options: {
            floor: 0,
            ceil: 100,
            step: 5,
            translate: function (value) {
                return value / 100;
            },
            showSelectionBar: true,
            getSelectionBarColor: function (value) {
                return 'black';
            }
        }
    }
    $scope.modelInfo = function () {
		//$http.get("/skincare/ModelInfo")
		$http.get($window.services.modelinfo)
		.success(function (response) {
			console.log("ModelInfo: " , response);
			$scope.modelInfo = response;
		})
	}
	$scope.modelInfo();

    $scope.showOverlay = function (file) {
        return $scope.hoverOn
        && file.abcd != undefined
        && file.abcd.largeimage != undefined
        && file.abcd.largeimage != ''
        //&& file.abcd.largeimage != file.abcd.segments
        && file.abcd.largeimage!=file.abcd.colors
        && file.abcd.largeimage!=file.abcd.asymvert
        && file.abcd.largeimage!=file.abcd.asymhoriz
        }

    $scope.maybeOverlay = function (file) {
        var ret =
           file.abcd != undefined
        && file.abcd.largeimage != undefined
        && file.abcd.largeimage != ''
        && (file.abcd.largeimage == file.abcd.border
            || (file.featureImages
               &&
            (file.abcd.largeimage == file.featureImages['segmentation']
           || file.abcd.largeimage == file.featureImages['pigment_network']
           || file.abcd.largeimage == file.featureImages['globules']
           || file.abcd.largeimage == file.featureImages['streaks']
           || file.abcd.largeimage == file.featureImages['milia_like_cyst']
           || file.abcd.largeimage == file.featureImages['negative_network']
           )));
        //console.log("May be " + ret, file.abcd ,file.abcd.largeimage )
    return ret;
        }

    $scope.showSample = function () {
        $scope.testWithSamples = true;
    }

    $scope.classify = function(file) {
		files = [file];

		console.log('classify    files ...', files);
		$scope.appendEvaluation = false;
		if (files !== undefined) {
			console.log("climg " , $scope.classifiedImages );

			recClassify();

		} else {
			$scope.uploadStatus = 'error';
			$scope.uploadedFile.answer = 'No File selected';
		}
	};

	function recClassify () {
		console.log("rec " , files, $scope.classifiedImages);
		var classifierUrl = $window.services.classifier;
		//var classifierUrl = "/skincare/Classifier";

		if (files.length > 0) {
			// get next and delete from list
			var current = files.shift();
			//$scope.classifiedImages[current.name].predict = ["...", "..."];
			classifyImage(current, classifierUrl);
			//$scope.evaluateResults()
		} else {
			$scope.classifierRunning = false;
			console.log("done, result: " , $scope.classifiedImages);
			console.log(" results: " , $scope.classificationResults);
		}
	}


	function classifyImage (file, classifierUrl) {
		var fd = new FormData();
		fd.append('file', file._file);

		console.log("current file ", file);
		$scope.classifierRunning = true;

        $scope.classifiedImages[file.name].predict = [];
        $scope.classifiedImages[file.name].class = {gifcompute: $scope.loadingGif};

		$http.post(classifierUrl, fd, {
			transformRequest: angular.identity,
			headers: {'Content-Type': undefined}
		})
		.success(function(response) {
			//console.log("response ", r, r.filename,                			r.prediction.replace(" ", ", "));
			var pred = {};
			if (response.prediction != undefined) {
				console.log("Success ", response);
				pred.name = response.filename;
				//pred.predict = JSON.parse(response.prediction.replace(" ", ", "));
				//$scope.classifiedImages[file.name].predict = JSON.parse(response.prediction);
				$scope.classifiedImages[file.name].predict = response.prediction;
				//$scope.classificationResults.push(pred.predict);
				console.log("upd climg " , $scope.classifiedImages);
			} else {
				$scope.classifiedImages[file.name].errormessage= "Classify: " + response.error;
				console.log(response.filename + " Error " , $scope.classifiedImages[file.name].error);
			}
            $scope.classifiedImages[file.name].class.gifcompute = null;
			recClassify();

			//return pred;
		})
		.error(function(response) {
			console.log(file.name + " Error " , response);
		});
	}

    $scope.segment = function (file) {
        //file = $scope.myFiles[0];
        //console.log("segment " , file);
		//if ($scope.classifiedImages[file.name] == undefined)
		 //   $scope.classifiedImages[file.name]= {name: file.name,  url: file.url, prediction: "..."};

        var fd = new FormData();
        fd.append('file', file._file);
        $http.post($window.services.segmentation, fd, {
        //$http.post('/skincare/Segment', fd, {
                transformRequest: angular.identity,
                responseType: "blob", // ??
                headers: {'Content-Type': undefined} // blob?
             })
        .success (function (response) {
            //console.log("Segment result ", response);
            //$scope.segmentationImage = URL.createObjectURL(response);
            file.featureImages['segmentation'] = URL.createObjectURL(response);
            //console.log("Segmentation Image ",file.featureImages['segmentation'])
        })
        .error (function (response) {
			$scope.classifiedImages[file.name].errormessage=response;
            console.log("Segment result ", response);
            })

        }

    $scope.allfeat = ['globules', 'streaks', 'pigment_network', 'milia_like_cyst', 'negative_network'];

    $scope.tryGetImageFeatures = function (file) {
    // check with one feature if image is ok
    // if response has no error continue with responseType "blob"
        var testfeature = "globules";
        var fd = new FormData();
        fd.append('file', file._file);
        file.giffeaturesrunning = $scope.loadingGif;

        //$http.post('/skincare/ExtractFeature/'
		$http.post($window.services.features
				+testfeature, fd, {
                transformRequest: angular.identity,
                //responseType: "blob", // ??
                headers: {'Content-Type': undefined} // blob?
             })
        .success (function (response) {
         if (response.error != undefined) {
            ckeckOk = false;
            console.log("Errors with image ", response);
            file.errormessage = "DL Features: " + response.error;
            file.giffeaturesrunning = '';
         } else {
			console.log("image ok, get features");
			
			
			// test ok (no error message): go and get the features (as blobs)			
            $scope.allFeatures(file);
            
         }
        })
        .error (function (response) {
            console.log("Error: Feature result "  , response);
            })
    }
    $scope.allFeatures = function (file) {
        if (file.featureImages == undefined)
           file.featureImages = [];
        $scope.segment(file);

         $scope.allfeat.forEach(function (f) {
            $scope.features(file, f);});
         setTimeout(function(){
             $scope.allfeaturesset(file); }, 1500);
    }


    $scope.allfeaturesset =function (file) {
        console.log("check all done");
        var alldone = true;
        $scope.allfeat.forEach(
         function (f) {
            //console.log("AF set? " + f, (file.featureImages[f]!=undefined));
            if (file.featureImages[f] == undefined)
             alldone = false;
             });
         if (!alldone) {
            //console.log("restart timer");
            if (true) setTimeout(function(){
             $scope.allfeaturesset(file); }, 1000);
         } else {
             //console.log("all set");
             file.giffeaturesrunning = '';
             $scope.$apply();
         }
     }

    $scope.checkImageForFeatures = function (file) {
        if (file.featureImages == undefined)
           file.featureImages = [];
        var ckeckOk = false;
        var fd = new FormData();
        //console.log("Features of ", file);
        fd.append('file', file._file);
        var feature = "globules";

        //$http.post('/skincare/ExtractFeature/'
		$http.post($window.services.features
				+feature, fd, {
                transformRequest: angular.identity,
                //responseType: "blob", // ??
                headers: {'Content-Type': undefined} // blob?
             })
        .success (function (response) {
         console.log("Feature " + feature , response);
         if (response.error != undefined) {
            ckeckOk = false;
            console.log("Feature result "+ feature , response);
            //$scope.featureImages[feature] = URL.createObjectURL(response);
            //file.featureImages[feature] = URL.createObjectURL(response);
            //console.log("Feature " + feature + " Image ",file.featureImages[feature])
         } else {
			console.log("Response Error: " + feature , response);
            ckeckOk = true;
         }
        })
        .error (function (response) {
            ckeckOk = false;
            console.log("Error: Feature result "  , response);
            })

        return ckeckOk;
    }
    $scope.features = function (file, features) {
        if (file.featureImages == undefined)
           file.featureImages = [];
        var fd = new FormData();
        //console.log("Features of ", file);
        fd.append('file', file._file);
        if (features == undefined || features.length == 0) {
        } else {
        var feature = features;

        //$http.post('/skincare/ExtractFeature/'
		$http.post($window.services.features
				+feature, fd, {
                transformRequest: angular.identity,
                responseType: "blob", // ??
                headers: {'Content-Type': undefined} // blob?
             })
        .success (function (response) {
         console.log("Feature " + feature , response);
         if (response.error != "") {
            console.log("Feature result "+ feature , response);
            //$scope.featureImages[feature] = URL.createObjectURL(response);
            file.featureImages[feature] = URL.createObjectURL(response);
            //console.log("Feature " + feature + " Image ",file.featureImages[feature])
         } else {
			file.errormessage=response.error;
            console.log("Response Error: " + feature , response);
         }
        })
        .error (function (response) {
			file.errormessage=response.error;
            console.log("Error: Feature result " + feature , response);
            })
        }
    }

    function getBase64ImgType (base64_image) {
    	switch (base64_image.charAt(0)) {
    	case '/' : return "jpg"; break;
    	case 'i' : return "png"; break;
    	case 'R' : return "gif"; break;
    	case 'U' : return "webp"; break;
    	default: return null;
    	}
    }
    function decodeImage (base64_image) {    	
    	return URL.createObjectURL(base64_2_blob(base64_image))
	}
    function base64_2_blob (base64_image) {
		const byteCharacters = atob(base64_image);
		const byteNumbers = new Array(byteCharacters.length);
		for (let i = 0; i < byteCharacters.length; i++) {
		    byteNumbers[i] = byteCharacters.charCodeAt(i);
		}
		const byteArray = new Uint8Array(byteNumbers);
		var type64 = getBase64ImgType(base64_image);
		const blob = new Blob([byteArray], {type: "image/" +type64});
		return blob;
	}
    $scope.abcd = function (file) {
        //file = $scope.myFiles[0];
        console.log("abcd " , file, );
		//if (file == undefined)
		 //   file= {name: file.name,  url: file.url, prediction: "..."};

        var fd = new FormData();
        fd.append('file', file._file, file.name);
        //fd.append('fct', 'abcd');

        file.abcd = {predict: ""};
        file.abcd.gifcompute = $scope.loadingGif;
        file.featureImages = {}
        file.featureImages["segments"] = $scope.loadingGif;
        file.featureImages["colors"] = $scope.loadingGif;
        file.featureImages["asymhoriz"] = $scope.loadingGif;
        file.featureImages["asymvert"] = $scope.loadingGif;
        
        //$http.post('/skincare/abcd',
		$http.post($window.services.abcd,
		//$http.post('/abcd',
				fd, {
                transformRequest: angular.identity,
                headers: {'Content-Type': undefined}
             })
        .success (function (response) {
        	try {
        		console.log("abcd result", response, response.images.length);
            
			file.abcd["colors"]   = decodeImage(response.images["colors"]);
            file.abcd["segments"] = decodeImage(response.images["segmented"]);
            file.abcd["asymhoriz"] = decodeImage(response.images["asymhoriz"]);
            file.abcd["asymvert"] = decodeImage(response.images["asymvert"]);
            file.abcd["border"] = decodeImage(response.images["border"]);
            file.featureImages["border"] = decodeImage(response.images["border"]);
            file.featureImages["segments"] 	= decodeImage(response.images["segmented"]);
            file.featureImages["colors"] 	= decodeImage(response.images["colors"]);
            file.featureImages["asymhoriz"] = decodeImage(response.images["asymhoriz"]);
            file.featureImages["asymvert"] 	= decodeImage(response.images["asymvert"]);
        		 	
            
    		file.abcd.predictvalues = response.features;
            /*
            Border: 0-10  max 1350
            Dia:    0-40
            Asym :  0-1     max 4.5
            Ziel: range 0-10
            */
            var factor = 0.9;

            var borderFactor = 10;
            var diaFactor = 10;
            var diaNorm = 0.25;
            var asymFactor = 10;
            var asymNorm = 10;

            var borderVal = Math.min(response.features['B'], 10);
            var dia1Val = Math.min(response.features['D1'], 40) * diaNorm;
            var dia2Val = Math.min(response.features['D2'], 40) * diaNorm;
            var asymHVal = Math.min(response.features['A1'], 1) * asymNorm;
            var asymVVal = Math.min(response.features['A2'], 1) * asymNorm;
            file.abcd.borderVal   = borderVal;
            file.abcd.borderPlus  = response.features['B'] > 10 ? ">" : "";
            file.abcd.dia1Val     = dia1Val;
            file.abcd.dia1Plus    = response.features['D1'] > 40 ? ">" : "";
            file.abcd.dia2Val     = dia2Val;
            file.abcd.dia1Plus    = response.features['D2'] > 40 ? ">" : "";
            file.abcd.asymHVal     = asymHVal;
            file.abcd.asymHPlus    = response.features['A1'] > 1 ? ">" : "";
            file.abcd.asymVVal     = asymVVal;
            file.abcd.asymVPlus    = response.features['A2'] > 1 ? ">" : "";

            file.abcd.borderStyleW =  {'width': borderVal * borderFactor * factor +"px"};
            file.abcd.dia1StyleW =    {'width': dia1Val * diaFactor * factor +"px"};
            file.abcd.dia2StyleW =    {'width': dia2Val * diaFactor * factor +"px"};
            file.abcd.aSymHStyleW =   {'width': asymHVal * asymFactor * factor +"px"};
            file.abcd.aSymVStyleW =   {'width': asymVVal * asymFactor * factor +"px"};

            file.abcd.centroids = abcdFeatures(file.abcd.predictvalues);
            file.abcd.predict =
                file.abcd.predictvalues['PRED'] == 1 ? 'malignant' : 'benign';
        	}
        	catch (e) {
            	file.abcd.gifcompute = null;
            	console.log("error ", e.message);
    			file.errormessage = e.message;
        	}
        	file.abcd.gifcompute = null;
        })
        .error (function(response) {
        	console.log("error ", response)
			file.errormessage = "Error computing ABCD features, response='" + response
					+ "' see also console output (probably blocked by CORS policy)";
			file.abcd.gifcompute = null;
			}
        )
    }
    $scope._abcd = function (file) {
        //file = $scope.myFiles[0];
        console.log("abcd " , file);
		//if ($scope.classifiedImages[file.name] == undefined)
		 //   $scope.classifiedImages[file.name]= {name: file.name,  url: file.url, prediction: "..."};

        var fd = new FormData();
        fd.append('images', file._file);
        fd.append('fct', 'abcd');

        $scope.classifiedImages[file.name].abcd = {predict: ""};
        $scope.classifiedImages[file.name].abcd.gifcompute = $scope.loadingGif;

        var skinroot = "/skinroot/";
        $http.post(skinroot + 'tools.php', fd, {
                transformRequest: angular.identity,
                headers: {'Content-Type': undefined}
             })
        .success (function (response) {
            var path = decodeURI(response.path);
            var filename = response.localfile;
            var abcdRoot = skinroot + path + "/" + filename;
            console.log("abcd result", path, filename);
            //$scope.classifiedImages[file.name].abcd = {"localfile": filename, "path": path};
            $scope.classifiedImages[file.name].abcd["colors"]   = abcdRoot + "_colors.PNG";
            $scope.classifiedImages[file.name].abcd["segments"] = abcdRoot + "_segmented.PNG";
            $scope.classifiedImages[file.name].abcd["contours"] = abcdRoot + ".PNG";
            $scope.classifiedImages[file.name].abcd["mask"] = abcdRoot + "_mask.PNG";
            $scope.classifiedImages[file.name].abcd["border"] = abcdRoot + "_active_contour.PNG";
            $scope.classifiedImages[file.name].abcd["asymhoriz"] = abcdRoot + "_horizontal.PNG";
            $scope.classifiedImages[file.name].abcd["asymvert"] = abcdRoot + "_vertical.PNG";
            $scope.classifiedImages[file.name].abcd["warped"] = abcdRoot + "_warped.PNG";

            $scope.classifiedImages[file.name].abcd.predictvalues = response.values;
            /*
            Border: 0-10  max 1350
            Dia:    0-40
            Asym :  0-1     max 4.5
            Ziel: range 0-10
            */
            var factor = 0.9;

            var borderFactor = 10;
            var diaFactor = 10;
            var diaNorm = 0.25;
            var asymFactor = 10;
            var asymNorm = 10;

            var borderVal = Math.min(response.values['B'], 10);
            var dia1Val = Math.min(response.values['D1'], 40) * diaNorm;
            var dia2Val = Math.min(response.values['D2'], 40) * diaNorm;
            var asymHVal = Math.min(response.values['A1'], 1) * asymNorm;
            var asymVVal = Math.min(response.values['A2'], 1) * asymNorm;
            $scope.classifiedImages[file.name].abcd.borderVal   = borderVal;
            $scope.classifiedImages[file.name].abcd.borderPlus  = response.values['B'] > 10 ? ">" : "";
            $scope.classifiedImages[file.name].abcd.dia1Val     = dia1Val;
            $scope.classifiedImages[file.name].abcd.dia1Plus    = response.values['D1'] > 40 ? ">" : "";
            $scope.classifiedImages[file.name].abcd.dia2Val     = dia2Val;
            $scope.classifiedImages[file.name].abcd.dia1Plus    = response.values['D2'] > 40 ? ">" : "";
            $scope.classifiedImages[file.name].abcd.asymHVal     = asymHVal;
            $scope.classifiedImages[file.name].abcd.asymHPlus    = response.values['A1'] > 1 ? ">" : "";
            $scope.classifiedImages[file.name].abcd.asymVVal     = asymVVal;
            $scope.classifiedImages[file.name].abcd.asymVPlus    = response.values['A2'] > 1 ? ">" : "";

            $scope.classifiedImages[file.name].abcd.borderStyleW =  {'width': borderVal * borderFactor * factor +"px"};
            $scope.classifiedImages[file.name].abcd.dia1StyleW =    {'width': dia1Val * diaFactor * factor +"px"};
            $scope.classifiedImages[file.name].abcd.dia2StyleW =    {'width': dia2Val * diaFactor * factor +"px"};
            $scope.classifiedImages[file.name].abcd.aSymHStyleW =   {'width': asymHVal * asymFactor * factor +"px"};
            $scope.classifiedImages[file.name].abcd.aSymVStyleW =   {'width': asymVVal * asymFactor * factor +"px"};

            $scope.classifiedImages[file.name].abcd.centroids = abcdFeatures($scope.classifiedImages[file.name].abcd.predictvalues);
            $scope.classifiedImages[file.name].abcd.predict =
                $scope.classifiedImages[file.name].abcd.predictvalues['PRED'] == 1 ? 'malignant' : 'benign';
            $scope.classifiedImages[file.name].abcd.gifcompute = null;
            $scope.classifiedImages[file.name].abcd.overColorImg = false;

        })
        .error (function(response) {
				$scope.classifiedImages[file.name].errormessage=response;}
        )
    }
    $scope.resetLargeImg = function (file) {
        if (!$scope.hoverOn) file.abcd.largeimage=file.url;
        file.abcd.mouseXY = '';
    }

    $scope.keydownOnImage = function (event, image) {
        console.log("Keydown ", event);
    }

    $scope.mouseOverImage1 = function (event, image) {

        var largeimg = image.abcd.largeimage;
        var colorimg = image.abcd.colors;
        //console.log("currentImg ", $scope.currentImg);
        //console.log("largeimg " , largeimg.src.substring(largeimg.src.lastIndexOf("/")+1), colorImgSrc.substring(colorImgSrc.lastIndexOf("\\")+1));
        //console.log("largeimg " , largeimg, colorimg, event);
        var curlarge= event.path[0];
        var srcLarge = largeimg.substring(largeimg.lastIndexOf("\\")+1);
        if (colorimg != undefined) {
            var srcColor = colorimg.substring(colorimg.lastIndexOf("\\")+1);
            //console.log("Mouse " + srcLarge + " " + srcColor + " show " + image.abcd.overColorImg)
            if (srcLarge && srcLarge == srcColor && image.abcd.overColorImg){
                image.abcd.mouseXY = Math.round(event.offsetX / (curlarge.width  / curlarge.naturalWidth))
                         + "," + Math.round(event.offsetY / (curlarge.height / curlarge.naturalHeight));
                         //console.log("XY " + image.abcd.mouseXY)
            } else {
                image.abcd.mouseXY = "";
                //console.log("reset XY " + image.abcd.mouseXY)
            }
            } else {
                image.abcd.mouseXY = "";
                //console.log("reset XY " + image.abcd.mouseXY)
                }
    }

    $scope.getImageList = function(type) {
        var images = [];
        var i = 1;
        var no = 10;
        var types = [];
        if (type == undefined)
            types = ['benign', 'malign'];

        types.forEach(function(type){
          if ($scope.testWithSamples) {

            $window.classifyTestImages.forEach(function(img) {
                if (i < no && img.type == type) {
                    img.url = "images/samples/" + img.name;
                    images.push(img);
                    i++;
                }
            });
          } else {
            Object.keys($scope.classifiedImages).forEach(function(key) {
                if (images.indexOf($scope.classifiedImages[key]) < 0)
                    images.push($scope.classifiedImages[key]);
            })};
        })
      return images;
    }

    $scope.xxgetImageList = function() {
        var images = [];
        Object.keys($scope.classifiedImages).forEach(function(key) {
            images.push($scope.classifiedImages[key]);
            })
           return images;
    }



    function abcdFeatures (predictvalues) {
        var allcolors = {
        'Blue Gray':    ['A_BG', 'green'],
        'White':        ['A_W', 'cyan'],
        'Light Brown':  ['A_LB', 'yellow'],
        'Dark Brown':   ['A_DB', 'red'],
        'Black':        ['A_B', 'black']
        };
        var colors_attr = predictvalues.colors_attr;
        var colors = [];
        colors_attr.forEach(function (c) {
            var cDef = allcolors[c.color];
            var cCtrds = c.centroids;
            var cVal = predictvalues[cDef[0]];
            colors.push({'color': c.color, 'centroids': cCtrds, 'val': cVal, 'line': cDef[1]});
            })
        return colors;
        //Object.keys(predictvalues).forEach(function (k,v) {
            //console.log("K " + k + " V " + v);


    }

	$scope.$watch('myFiles',
	function (newval, oldval) {
        if (newval != oldval) {
         console.log("Add to Files: " , newval);
         if (newval != null && newval.length > 0) {
            newval.map(function(i){
                $scope.classifiedImages[i.name] = 
                {index: $scope.imgIdx, name: i.name, url: i.url, _file: i._file};
            });
            $scope.imgIdx++;
            //$scope.myFiles   = [];
         }
       }
	});

	$scope.$watch('classifierRunning',
			function (newval, oldval) {
			if (newval == oldval || newval == null) return;
			if (newval == true) {
				// set cursor to progress
				$scope.bodyStyle={cursor: 'progress'};
			} else {
				// set cursor default
				$scope.bodyStyle={cursor: 'default'};
			}
			console.log("classifierRunning: ", newval);
	});
    $scope.$watch('opaSlider.value',
        function (newval, oldval) {
            if (newval != oldval) {
                $scope.hoverStyle = {opacity: (newval/100)};
                }
    })

    $scope.$watch('keyIsDown',
	function (newval, oldval) {
        if (newval != oldval) {
        console.log(" shift key pressed: " + newval)
        }
        });

    $scope.$watch('hoverOnPac',
	function (newval, oldval) {
        if (newval != oldval) {
        console.log(" hover On: " + newval)
        }
        });

    $scope.toggleHoverOn = function () {
        $scope.hoverOn = !$scope.hoverOn;
    }


}]);

