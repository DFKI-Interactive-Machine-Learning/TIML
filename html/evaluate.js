var myApp = angular.module('myApp', []);

myApp.config(function ($httpProvider){
       	  //$httpProvider.defaults.transformRequest.unshift($httpParamSerializerJQLikeProvider.$get());
       	  $httpProvider.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8';
       	});


myApp.directive('ngFileModel', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function (scope, element, attrs) {
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

myApp.directive('upload', ['$parse', function($parse) {
    return {
      restrict: 'EA',
      replace: true,
      //scope: {},
      //require: '?ngModel',
      //template: '<div class="asset-upload">Drop files here</div>',
      link: function(scope, element, attrs, ngModel) {
        //console.log("Event" , event);
        var model = $parse(attrs.ngFileModel);
        var isMultiple = attrs.multiple=="true";
        var modelSetter = model.assign;
        element.on('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
        element.on('dragenter', function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
        element.on('drop', function(e) {
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
                };

            return false;
        });

        }
    };
}]);


myApp.controller('myCtrl', ['$scope', '$http', '$sce', '$window', /*'fileUpload',*/ 


function($scope, $http, $sce, $window, fileUpload) 
{
	
	$scope.version="0.0.1a4";

	$scope.testWithSamples = false;

	$scope.uploadStatus = "";
	$scope.uploadedFile = {};
	$scope.classifiedImages = {benign: [], malign: []};
	$scope.classificationResults = {benign: [], malign: []};
	$scope.imgIdx = 0;
	$scope.appendEvaluation = false;
    $scope.loadingGif = "loadingwedges.gif";
    $scope.markAsBadThreshold = 0.9;
	
	var files = [];
	$scope.classifierRunning = false;

	$scope.classify_evaluate = function () {
		files = $scope.getImageList();
		console.log('classify    files ...', files);
		
		$scope.appendEvaluation = true;
		recClassify();	
	}
	
	$scope.classifySingle = function(file) {
		files = [file];

		console.log('classify  single file ...', files);
		$scope.appendEvaluation = false;
		if (files !== undefined) {
			console.log("climg " , $scope.classifiedImages );

			recClassify();

		} else {
			$scope.uploadStatus = 'error';
			$scope.uploadedFile.answer = 'No File selected';
		}
	};
	$scope.classifyAll = function(BorM) {

		/*var input = (BorM == 'benign') ? $scope.myFilesBenign : $scope.myFilesMalign;
		files = [];
		console.log("All " , $scope.classifiedImages)
		$scope.classifiedImages[BorM].forEach(function(item) {
			//item.type = BorM;
			console.log(" add " , item)
			files.push(item)
		});*/

        files = $scope.getImageList(BorM);

		console.log('classify    files ...' + BorM, files);
		$scope.appendEvaluation = false;
		if (files !== undefined) {
			console.log("climg " , files );
			
			recClassify();
			
		} else {
			$scope.uploadStatus = 'error';
			$scope.uploadedFile.answer = 'No File selected';
		}
	};
	
	function recClassify () {   
		console.log("rec " , files, $scope.classifiedImages);
		var classifierUrl = "/skincare/Classifier";
		
		if (files.length > 0) {
			// get next and delete from list
			var current = files.shift();
			//$scope.classifiedImages[current.type][current.name].predict = ["...", "..."];
			if ($scope.classifiedImages[current.type][current.name].predict) {
			    console.log("Skip " , current);
			    recClassify();
			}
			else
			    classifyImage(current, classifierUrl);

		} else {
			$scope.classifierRunning = false;
			console.log("done, result: " , $scope.classifiedImages);
			console.log(" results: " , $scope.classificationResults);
			
			if ($scope.appendEvaluation == true){
				$scope.evaluateResults();
				$scope.appendEvaluation == false;
			}
		}
	}

	$scope.modelInfo = function () {
		$http.get("/skincare/ModelInfo")
		.success(function (response) {
			console.log("ModelInfo: " , response);
			$scope.modelInfo = response;
		})
	}
	$scope.modelInfo();
	
	function classifyImage (file, classifierUrl) {
		var fd = new FormData();
		fd.append('file', file._file);

		console.log("current file ", file);
		$scope.classifierRunning = true;
		$scope.classifiedImages[file.type][file.name].class = {gifcompute: $scope.loadingGif};
        $scope.classifiedImages[file.type][file.name].predict = [];
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
				var predict = JSON.parse(response.prediction); //.replace(" ", ", "));
				$scope.classifiedImages[file.type][pred.name].predict = predict;
				$scope.classificationResults[file.type].push(predict);
				console.log("upd climg " , $scope.classifiedImages[file.type]);
			} else {
				console.log(r.filename + " Error " , r.error);
			}
			$scope.classifiedImages[file.type][pred.name].class.gifcompute = null;
			recClassify();
			
			return pred;
		})
		.error(function(response) {
			console.log(file.name + " Error " , response);
		});
	}
	
	
	$scope.evaluateResults = function (test) {
		// all curves
		
		var chartOptions = $scope.chartOptionsMain;
		chartOptions.chart.renderTo = "chart";
		var chart = 
			new Highcharts.Chart(chartOptions);
		
		for (var i=0; i < chart.series.length; i++) {
			var ser = chart.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}

		// ROC
		var rocOptions = $scope.rocOptions;
		rocOptions.chart.renderTo = "roc";
		var chartROC = 
			new Highcharts.Chart(rocOptions);
		
		for (var i=0; i < chartROC.series.length; i++) {
			var ser = chartROC.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}

		// Spec/sens
		var prOptions = $scope.prOptions;
		prOptions.chart.renderTo = "precrec";
		var chartPR = 
			new Highcharts.Chart(prOptions);
		
		for (var i=0; i < chartPR.series.length; i++) {
			var ser = chartPR.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}
		
		// if test use stored results (modelresults.js)
		var singleResults = test ? $window.latestmodel : $scope.classificationResults;
        $scope.testWithSamples = test;

		console.log(singleResults);
		//console.log("chart series", chart.series);
		var distance = 0.005;
		var ts = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
				  0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
		//for (thres = 0.005; thres < 1.0; thres+=distance){
		ts.forEach(function(thres) {
			
			var res = computeQuality(singleResults, thres, distance);
			if(false)
				console.log(" t: " + roundTo(thres,2) + " tp: " + res.tp + " fn: " + res.fn 
					+ " tn: " + res.tn + " fp: " + res.fp );
			chart.get("precision").addPoint([roundTo(thres,2), roundTo(res.precision, 3)]);
			chart.get("recall")	  .addPoint([roundTo(thres,2), roundTo(res.recall, 3)]);
			chart.get("accuracy") .addPoint([roundTo(thres,2), roundTo(res.accuracy, 3)]);
			chart.get("f1")		  .addPoint([roundTo(thres,2), roundTo(res.f1,3)]);
			chart.get("spec")	  .addPoint([roundTo(thres,2), roundTo(res.spec,3)]);
			
			chartROC.get("roc")		.addPoint([roundTo(res.fpr,2), roundTo(res.tpr,3)]);
			
			chartPR.get("prec")	.addPoint([roundTo(res.recall,2), roundTo(res.precision,3)]);
			
			//chartPN.get("malign")	.addPoint([roundTo(thres,2), roundTo(res.pos,3)]);
			//chartPN.get("benign")	.addPoint([roundTo(thres,2), roundTo(res.neg,3)]);
		});
		chartROC.get("rocmean")	.addPoint([roundTo(0,2), roundTo(0,3)]);
		chartROC.get("rocmean")	.addPoint([roundTo(1,2), roundTo(1,3)]);
		//console.log("ROC ");
		//chartROC.series[0].data.map(function(d){console.log(d.x + " " + d.y)});
		chart.setSize(600,250, true);
		
		chartROC.setSize(300,250, true);
		chartPR.setSize(300,250, true);

		//console.log("Evaluation: " , res);
		//$scope.testWithSamples = false;

	}
	
	$scope.evaluateResultsX = function (test) {
		// all curves
		/*
		var chartOptions = $scope.chartOptions1;
		chartOptions.chart.renderTo = "chart";
		var chart = 
			new Highcharts.Chart(chartOptions);
		
		var noseries = chart.series.length;
		for (var i=0; i < chart.series.length; i++) {
			var ser = chart.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}
		// ROC
		var rocOptions = $scope.rocOptions;
		rocOptions.chart.renderTo = "roc";
		var chartROC = 
			new Highcharts.Chart(rocOptions);
		
		for (var i=0; i < chartROC.series.length; i++) {
			var ser = chartROC.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}

		// Spec/sens
		var prOptions = $scope.prOptions;
		prOptions.chart.renderTo = "precrec";
		var chartPR = 
			new Highcharts.Chart(prOptions);
		
		for (var i=0; i < chartPR.series.length; i++) {
			var ser = chartPR.series[i];
			while (ser.data.length > 0) {
				ser.data[0].remove(false);
			}
		}
		 */
		clearCharts();
		
		var singleResults = test ? $window.latestmodel : $scope.classificationResults;
		console.log(singleResults);
		//console.log("chart series", chart.series);
		var distance = 0.025;
		var resultsx = [];
		var ts = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
			  0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
		var ts = [0.9, 0.85, 0.8, 0.75, 0.7, 0.675, 0.65, 0.625, 0.6, 0.5, 0.4, 0.375, 0.35, 0.325, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1];
	//for (thres = 0.005; thres < 1.0; thres+=distance){
		ts.forEach(function(thres) {
		

			var res = {s_tp:0, s_tn:0, s_fp:0, s_fn:0};
			
			for (i=0; i < 20; i++) {
				var res1 = computeQualityX(singleResults, thres, distance);
				res.s_tp += 1.0 * res1.tp/res1.ms;
				res.s_fn += 1.0 * res1.fn/res1.ms;
				res.s_tn += 1.0 * res1.tn/res1.bs;
				res.s_fp += 1.0 * res1.fp/res1.bs;
				};
			res.precision 	= 1.0 * res.s_tp / (res.s_tp + res.s_fp);
			res.recall 	 	= 1.0 * res.s_tp / (res.s_tp + res.s_fn);
			res.accuracy  	= 1.0 * (res.s_tp + res.s_tn) / (res.s_tp + res.s_tn + res.s_fp + res.s_fn);
			res.f1		 	= 1.0 * 2*res.s_tp / (2*res.s_tp + res.s_fp + res.s_fn);
			res.spec		= 1.0 * res.s_tn / (res.s_tn + res.s_fp);
			
			res.tpr			= 1.0 * res.s_tp / (res.s_tp + res.s_fn);
			res.fpr			= 1.0 * res.s_fp / (res.s_tn + res.s_fp);	
			if(true)
				console.log(" -> TH: " + roundTo(thres,2) + " fpr: " + res.fpr + " tpr: " + res.tpr + 
						" tp: " + roundTo(res.s_tp,2)  + " fn: " + roundTo(res.s_fn,2) 
						+ " tn: " + roundTo(res.s_tn,2) + " fp: " + roundTo(res.s_fp,2) );
	
			
			chart.get("precision").addPoint([roundTo(thres,3), roundTo(res.precision, 3)]);
			chart.get("recall")	  .addPoint([roundTo(thres,3), roundTo(res.recall, 3)]);
			chart.get("accuracy") .addPoint([roundTo(thres,3), roundTo(res.accuracy, 3)]);
			chart.get("f1")		  .addPoint([roundTo(thres,3), roundTo(res.f1,3)]);
			chart.get("spec")	  .addPoint([roundTo(thres,3), roundTo(res.spec,3)]);
			
			chart.get("fpr")	  .addPoint([roundTo(thres,3), roundTo(res.fpr,3)]);
			chart.get("tpr")	  .addPoint([roundTo(thres,3), roundTo(res.tpr,3)]);
			chartROC.get("roc")	  .addPoint([roundTo(res.fpr,3), roundTo(res.tpr,3)]);
			
			chartPR.get("prec")	  .addPoint([roundTo(res.recall,3), roundTo(res.precision,3)]);
			
			//chartPN.get("malign")	.addPoint([roundTo(thres,2), roundTo(res.pos,3)]);
			//chartPN.get("benign")	.addPoint([roundTo(thres,2), roundTo(res.neg,3)]);
		});
		console.log("ser roc", chartROC.get("roc").data);
		console.log("ser prec", chartPR.get("prec").data);
		chartROC.get("rocmean")	.addPoint([roundTo(0,2), roundTo(0,3)]);
		chartROC.get("rocmean")	.addPoint([roundTo(1,2), roundTo(1,3)]);
		console.log("ROC FPR/TPR");
		chartROC.series[0].data.map(function(d){console.log(d.x + " " + d.y)});
		chart.setSize(600,250, true);
		
		chartROC.setSize(300,250, true);
		chartPR.setSize(300,250, true);

		
	}
	function computeQualityX (classifications, thres, distance) {
		var res = {tp: 0, tn: 0, fp: 0, fn: 0, pos: 0, neg: 0, bs: 0, ms: 0};
		var no_ms = classifications['malign'].length;
		var no_bs = classifications['benign'].length;
		var n2p = no_ms / no_bs;
		var tot_img = no_ms + no_bs;

		var rands = [];
		var fpr = 0;
		while (res.bs < no_ms) {
			var idx = Math.floor(Math.random() * 100) % (no_bs/tot_img*100);
			// new element?
			if (rands.indexOf(idx) == -1) {
				rands.push(idx);
				var obj = classifications['benign'][idx];
				res.bs++;
				if (obj[1] <= thres) { 
					res.tn++; }
				else { 
					res.fp++; }
			}
			if (res.tn > 0 && res.fp > 0)
				fpr = 1.0 * res.fp / (res.tn + res.fp);		
			//console.log("FPR " + fpr + " rands " + rands);
		}
		res.fpr=fpr;
		//console.log("rands " , rands);
		
		classifications['malign'].forEach(function (obj) {
			if (obj[1] <= thres) { 
				res.fn++; }
			else { 
				res.tp++; }
			res.ms++;
			if (Math.abs(obj[1] - thres) < (distance*0.5)) {
				res.pos++;
			}
			})
		res.precision 	= 1.0 * res.tp / (res.tp + res.fp);
		res.recall 	 	= 1.0 * res.tp / (res.tp + res.fn);
		res.accuracy  	= 1.0 * (res.tp + res.tn) / (res.tp + res.tn + res.fp + res.fn);
		res.f1		 	= 1.0 * 2*res.tp / (2*res.tp + res.fp + res.fn);
		res.spec		= 1.0 * res.tn / (res.tn + res.fp);
		
		res.tpr			= 1.0 * res.tp / (res.tp + res.fn);
		res.fpr			= 1.0 * res.fp / (res.tn + res.fp);
		
		if (false)
		console.log("TH: " + roundTo(thres,2) + " fpr: " + res.fpr + " tpr: " + res.tpr + " tp: " + res.tp
				+ " fn: " + res.fn  + " tn: " +  res.tn + " fp: " + res.fp + " Bs: " + res.bs + " Ms: " + res.ms 
				+ " idx " + rands);
		return res;
	}
	function computeQualityXX (classifications, thres, distance) {
		var res = {tp: 0, tn: 0, fp: 0, fn: 0, pos: 0, neg: 0, bs: 0, ms: 0};
		var no_ms = classifications['malign'].length;
		var no_bs = classifications['benign'].length;
		var n2p = no_ms / no_bs;

		classifications['benign'].forEach(function (obj) {
		
			if (Math.random() < n2p && res.bs < no_ms) {  
			res.bs++;
			if (obj[1] <= thres) { 
				res.tn++; }
			else { 
				res.fp++; }
			if (Math.abs(obj[1] - thres) < (distance*0.5)) {
				res.neg++;
			}}
			})
		
		classifications['malign'].forEach(function (obj) {
			if (obj[1] <= thres) { 
				res.fn++; }
			else { 
				res.tp++; }
			res.ms++;
			if (Math.abs(obj[1] - thres) < (distance*0.5)) {
				res.pos++;
			}
			})
		res.precision 	= 1.0 * res.tp / (res.tp + res.fp);
		res.recall 	 	= 1.0 * res.tp / (res.tp + res.fn);
		res.accuracy  	= 1.0 * (res.tp + res.tn) / (res.tp + res.tn + res.fp + res.fn);
		res.f1		 	= 1.0 * 2*res.tp / (2*res.tp + res.fp + res.fn);
		res.spec		= 1.0 * res.tn / (res.tn + res.fp);
		
		res.tpr			= 1.0 * res.tp / (res.tp + res.fn);
		res.fpr			= 1.0 * res.fp / (res.tn + res.fp);
	
		if (true)
		console.log("t: " + roundTo(thres,2) + " tp: " + res.tp
				+ " fn: " + res.fn  + " tn: " + res.tn + " fp: " + res.fp + " Bs: " + res.bs + " Ms: " + res.ms );
		return res;
	}
	function computeQuality (classifications, thres, distance) {
		var res = {tp: 0, tn: 0, fp: 0, fn: 0, pos: 0, neg: 0};

		classifications['benign'].forEach(function (obj) {
		
			if (obj[1] <= thres) { 
				res.tn++; }
			else { 
				res.fp++; }
			
			if (Math.abs(obj[1] - thres) < (distance*0.5)) {
				res.neg++;
			}
			})
		
		classifications['malign'].forEach(function (obj) {
			if (obj[1] <= thres) { 
				res.fn++; }
			else { 
				res.tp++; }
			
			if (Math.abs(obj[1] - thres) < (distance*0.5)) {
				res.pos++;
			}
			})	
		//this.validationDataSet.positive = res.tp + res.fn;
		
		res.precision 	= 1.0 * res.tp / (res.tp + res.fp);
		res.recall 	 	= 1.0 * res.tp / (res.tp + res.fn);
		res.accuracy  	= 1.0 * (res.tp + res.tn) / (res.tp + res.tn + res.fp + res.fn);
		res.f1		 	= 1.0 * 2*res.tp / (2*res.tp + res.fp + res.fn);
		res.spec		= 1.0 * res.tn / (res.tn + res.fp);
		
		res.tpr			= 1.0 * res.tp / (res.tp + res.fn);
		res.fpr			= 1.0 * res.fp / (res.tn + res.fp);
		
		//res.pos			= res.pos / classifications['malign'].length;
		//res.neg			= res.neg / classifications['benign'].length;
		
		return res;
	}
	
	function roundTo (number, precision) {
		var factor = Math.pow(10, precision);
		  return Math.round(number * factor) / factor;
	}
    $scope.getImageList = function(type) {
        var images = [];
        var i = 1;
        var no = 10;
        if (type != undefined) {
            if ($scope.testWithSamples) {
              $window.imageListEx.forEach(function(img) {
                if (i <= no && img.type == type) {
                    img.url = "images/samples/Image_" + img.name;
                    images.push(img);
                    i++;
                }
              });
            } else {
              Object.keys($scope.classifiedImages[type]).forEach(function(key) {
                images.push($scope.classifiedImages[type][key]);
              })};
        } else {
          ['benign', 'malign'].forEach(function(type){
          if ($scope.testWithSamples) {
            $window.imageListEx.forEach(function(img) {
                if (i < no && img.type == type) {
                    img.url = "images/samples/" + img.name;
                    images.push(img);
                    i++;
                }
            });
          } else {
            Object.keys($scope.classifiedImages[type]).forEach(function(key) {
                images.push($scope.classifiedImages[type][key]);
            })};
        })}

           return images;
    }


	$scope.$watch('myFilesBenign',
	function (newval, oldval) {
        if (newval != oldval) {
         //console.log("benign Files: " , newval);
         if (newval != null && newval.length > 0) {
            newval.map(function(i){
                var obj = {index: $scope.imgIdx, type: 'benign', name: i.name, url: i.url, _file: i._file};
                console.log("add to benign " , obj)
                $scope.classifiedImages['benign'][i.name] = obj;
            });
            $scope.imgIdx++;
            //$scope.myFiles   = [];
         }
       }
	});
	$scope.$watch('myFilesMalign',
	function (newval, oldval) {
        if (newval != oldval) {
         //console.log("new malign Files: " , newval);
         if (newval != null && newval.length > 0) {
            newval.map(function(i){
                var obj = {index: $scope.imgIdx, type: 'malign', name: i.name, url: i.url, _file: i._file};
                console.log("add to malign " , obj)
                $scope.classifiedImages['malign'][i.name] = obj;
            });
            $scope.imgIdx++;
            //$scope.myFiles   = [];
                     }
       }
	});
    $scope.$watch('testWithSamples',
    function (newval, oldval) {
        if (newval != oldval) {
         console.log("testWithSamples: " + newval);
         }})

	$scope.$watch('XXclassifierRunning',
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
	
	
	$scope.chartOptionsMain = 
	{
		    chart: { 
		    	renderTo: "chart",
		    	//width:  1,
		    	//height: 1,
			    shadow: true,
			    backgroundColor: '#f1f1f1',
		    },
		    title: {       
		    	text: 'Evaluation'	    
		    },
		    xAxis: {
		    	title: {
		            text: 'Threshold'
		        },
		        tickInterval: 0.1,
		        startOnTick: false,
		    },
		    yAxis: {
		    	max: 1.0,
		        tickInterval: 0.1,
		    },
		    tooltip: {
		        shared: true,
		        crosshairs: true
		    },
		    legend: {
		        layout: 'vertical',
		        align: 'right',
		        verticalAlign: 'middle'
		    },
		    plotOptions: {
		    	series: {
		    		marker: {
		    			enabled: false
		    		}
		    	}
		    },
		    series: [{
		    		name: 'Precision',
		    		id: "precision",
		    		//color: "red"
		    	},
		    	{
		    		name: 'Recall',
		    		id: "recall",
		    		//color: "blue"
				},
				{
					name: 'Specificity',
					id: "spec",
					//color: "orange"
				},
				{
					name: 'Accuracy',
					id: "accuracy",
					//color: "green"
				},
				{
					name: 'F1',
					id: "f1",
					//color: "orange"
				},
		    	{
		    		name: 'FPR',
		    		id: "fpr",
		    		//color: "blue"
				},
		    	{
		    		name: 'TPR',
		    		id: "tpr",
		    		//color: "blue"
				},
				],
	}
	$scope.rocOptions = 
	{
		    chart: { 
		    	renderTo: "roc",
		    	//width:  1,
		    	//height: 1,
			    shadow: true,
			    backgroundColor: '#f1f1f1',
		    },
		    title: {       
		    	text: 'ROC'	    
		    },
		    xAxis: {
		    	title: {
		            text: 'FPR'
		        },
		        tickInterval: 0.1,
		        startOnTick: false,
		    },
		    yAxis: {
		    	title: {
		    		text: 'TPR'
		    	},
		    	max: 1.0,
		        tickInterval: 0.1,
		        gridLineColor: 'lightgrey',
		    },
		    tooltip: {
		        shared: true,
		        crosshairs: true
		    },
		    legend: {
		    	enabled: false,
		        layout: 'vertical',
		        align: 'right',
		        verticalAlign: 'middle'
		    },
		    plotOptions: {
		    	sseries: {
		    		marker: {
		    			enabled: false
		    		}
		    	},line: {
		            dataLabels: {
		                enabled: false
		            },
		            //enableMouseTracking: false
		        }
		    },
		    series: [{
		    		name: 'ROC',
		    		id: "roc",
		    		},
		    		{
		    		name: 'ROCmean',
		    		id: "rocmean",
		    		color: "lightgrey"
		    		}
				],
	}
	$scope.prOptions = 
	{
		    chart: { 
		    	renderTo: "precrec",
		    	//width:  1,
		    	//height: 1,
			    shadow: true,
			    backgroundColor: '#f1f1f1',
		    },
		    title: {       
		    	text: 'Precision / Recall'	    
		    },
		    xAxis: {
		    	title: {
		            text: 'Recall'
		        },
		        tickInterval: 0.1,
		        startOnTick: false,
		    },
		    yAxis: {
		    	title: {
		    		text: 'Precision'
		    	},
		    	min: 0.1,
		    	max: 1.0,
		        tickInterval: 0.1,
		        gridLineColor: 'lightgrey',
		    },
		    tooltip: {
		        shared: true,
		        crosshairs: true
		    },
		    legend: {
		    	enabled: false,
		        layout: 'vertical',
		        align: 'right',
		        verticalAlign: 'middle'
		    },
		    plotOptions: {
		    	series: {
		    		marker: {
		    			enabled: false
		    		}
		    	}
		    },
		    series: [{
		    		name: 'prec',
		    		id: "prec",
		    		//color: "red"
		    		}
				],
	}
	$scope.pnOptions = 
	{
		    chart: { 
		    	renderTo: "preneg",
		    	//width:  1,
		    	//height: 1,
			    shadow: true,
			    backgroundColor: '#f1f1f1',
		    },
		    title: {       
		    	text: 'Pos/neg'	    
		    },
		    xAxis: {
		    	title: {
		            text: 'malign'
		        },
		        tickInterval: 0.1,
		        startOnTick: false,
		    },
		    yAxis: {
		    	title: {
		    		text: '#'
		    	},
		    	max: 1.0,
		        tickInterval: 0.1,
		    },
		    tooltip: {
		        shared: true,
		        crosshairs: true
		    },
		    legend: {
		    	enabled: false,
		        layout: 'vertical',
		        align: 'right',
		        verticalAlign: 'middle'
		    },
		    plotOptions: {
		    	series: {
		    		marker: {
		    			enabled: false
		    		}
		    	}
		    },
		    series: [{
		    		name: 'malign',
		    		id: "malign",
		    		//color: "red"
		    		},
		    		{
			    		name: 'benign',
			    		id: "benign",
			    		//color: "red"
			    		}
				],
	}
}]);