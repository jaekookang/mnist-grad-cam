// Modified from https://github.com/sugyan/tensorflow-mnist
var clear_clicked = true

class Main {
	constructor() {
		this.canvas = document.getElementById('main');
		this.input = document.getElementById('input');
		this.cam = document.getElementById('cam');
		this.gcam = document.getElementById('guided-cam');
		this.log = document.getElementById('log');

		this.log.style.fontsize = '25px';

		this.canvas.width = 449; // 16*28 + 1
		this.canvas.height = 449; // 16*28 + 1
		this.ctx = this.canvas.getContext('2d');
		this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
		this.canvas.addEventListener('touchstart', this.onMouseDown.bind(this));
		this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
		this.canvas.addEventListener('touchend', this.onMouseUp.bind(this));
		this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));

		this.initialize();
	}

	initialize() {
		// Make a bounding box
		this.ctx.fillStyle = '#FFFFFF';
		this.ctx.fillRect(0, 0, 449, 449);
		this.ctx.lineWidth = 1;
		this.ctx.strokeRect(0, 0, 449, 449);
		this.ctx.lineWidth = 0.05;
		// Make a grid
		for (var i=0; i<27; i++) {
			// Add vertical grids
			this.ctx.beginPath();
			this.ctx.moveTo((i+1)*16, 0);
			this.ctx.lineTo((i+1)*16, 449);
			this.ctx.closePath();
			this.ctx.stroke();
			// Add horizontal grids
			this.ctx.beginPath();
			this.ctx.moveTo(0,   (i+1)*16);
			this.ctx.lineTo(449, (i+1)*16);
			this.ctx.closePath();
			this.ctx.stroke();
		}
		// Clear canvas
		var ctx_input = this.input.getContext('2d');
		var ctx_cam = this.cam.getContext('2d');
		var ctx_gcam = this.gcam.getContext('2d');

		ctx_input.clearRect(0, 0, 140, 140);
		ctx_cam.clearRect(0, 0, 140, 140);
		ctx_gcam.clearRect(0, 0, 140, 140);

		// Clear table
		for (let j=0; j<10; j++) {
			$('#output tr').eq(j+1).find('td').eq(0).text(j);
			$('#output tr').eq(j+1).find('td').eq(1).text('');
			$('#output tr').eq(j+1).find('td').eq(0).removeClass('success');
			$('#output tr').eq(j+1).find('td').eq(1).removeClass('success');
		}

		if (!clear_clicked) {
			this.drawInput();
		}
		
	}

	onMouseDown(e) {
		// When pressed => get real-time position
		var mouseX = e.clientX || e.pageX;
		var mouseY = e.clientY || e.pageY;
		this.canvas.style.cursor = 'default';
		this.drawing = true;
		this.prev = this.getPosition(mouseX, mouseY);
	}

	onTouchStart(e) {
		// When pressed => get real-time position
		this.canvas.style.cursor = 'default';
		this.drawing = true;
	}

	onMouseUp() {
		// When released => draw input
		this.drawing = false;
		this.drawInput();
		this.log.innerHTML = 'Calculating...';
	}

	onMouseMove(e) {
		if (this.drawing) {
			var mouseX = e.clientX || e.pageX;
			var mouseY = e.clientY || e.pageY;
			var curr = this.getPosition(mouseX, mouseY);
			this.ctx.lineWidth = 16;
			this.ctx.lineCap = 'round';
			this.ctx.beginPath();
			this.ctx.moveTo(this.prev.x, this.prev.y); // prev
			this.ctx.lineTo(curr.x, curr.y); // current
			this.ctx.stroke();
			this.ctx.closePath();
			this.prev = curr;
		}
	}

	getPosition(clientX, clientY) {
		var rect = this.canvas.getBoundingClientRect();
		return {
			x: clientX - rect.left,
			y: clientY - rect.top
		};
	}

	drawInput() {
		// Make canvas input as image
		var ctx = this.input.getContext('2d');
		var img = new Image();
		img.onload = () => {
			var inputs = [];
			var small = document.createElement('canvas').getContext('2d');
			// small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
			small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
			var data = small.getImageData(0, 0, 28, 28).data;
			// For each row
			for (var i=0; i<28; i++) {
				// For each column, draw the number
				for (var j=0; j<28; j++) {
					var n = 4*(i*28 + j); // (R,G,B),(R,G,B),(R,G,B),(R,G,B),...
					inputs[i*28 + j] = (data[n+0] + data[n+1] + data[n+2])/3; // avg(R,G,B)
					ctx.fillStyle = 'rgb(' + [data[n+0], data[n+1], data[n+2]].join(',') + ')';
					ctx.fillRect(j*5, i*5, 5, 5);
				}
			};
			if (Math.min(...inputs) > 250) {
				return;
			};
			$.ajax({
				url: '/mnist',
				method: 'POST',
				contentType: 'application/json',
				data: JSON.stringify(inputs), // data sent to the server
				success: (data) => {          // data retrieved from the server
					// console.log(data);
					// Add prediction into the table
					var max = 0;
					var max_index = 0;
					// For each prediction: 0, 1, 2,..., 9
					for (let j=0; j<10; j++) {
						var value = Math.round(data.results[0][j]*1000);
						if (value > max) {
							max = value;
							max_index = j;
						}
						var digits = String(value).length;
						for (var k=0; k<3-digits; k++) {
							value = '0' + value;
						}
						var text = '0.' + value;
						if (value > 999) {
							text = '1.000';
						}
						$('#output tr').eq(j+1).find('td').eq(1).text(text);
					}
					
					for (let j=0; j<10; j++) {
						if (j === max_index) {
							$('#output tr').eq(j+1).find('td').eq(0).addClass('success');
							$('#output tr').eq(j+1).find('td').eq(1).addClass('success');
						} else {
							$('#output tr').eq(j+1).find('td').eq(0).removeClass('success');
							$('#output tr').eq(j+1).find('td').eq(1).removeClass('success');
						}
					}

					// Make Grad-CAM image
					var ctx_cam = this.cam.getContext('2d');
					var img_cam = new Image();
					img_cam.src = 'static/outputs/gradcam.jpg';
					img_cam.onload = () => {
						ctx_cam.drawImage(img_cam, 1, 1, 140, 140);
					}

					// Make Guided Grad-CAM image
					var ctx_gcam = this.gcam.getContext('2d');
					var cvs_gcam = document.createElement('canvas');
					var img_gcam = new Image();
					img_gcam.src = 'static/outputs/guided_gradcam.jpg';
					img_gcam.onload = () => {
						ctx_gcam.drawImage(img_gcam, 1, 1, 140, 140);
					}
					this.log.innerHTML = 'Done';

				},
				error: (jqXHR) => {
					this.log.innerHTML = 'error:'+jqXHR.status;	
				}
			})
		};
		img.src = this.canvas.toDataURL();
		// fs.unlink('static/outputs/guided_gradcam.jpg');
	}
}
$(() => {
	var main = new Main();
	$('#clear').click(() => {
		clear_clicked = true;
		main.initialize();
		document.getElementById('log').innerHTML = '';
		clear_clicked = false;
	});
});
