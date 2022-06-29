openSocket = () => {

    socket = new WebSocket("ws://192.168.84.117:9997/"); //ip jetson
    // socket = new WebSocket("ws://192.168.0.108:9997/"); //ip local
    let msg = document.getElementById("msg");
    socket.addEventListener('open', (e) => {
        document.getElementById("status").innerHTML = "Opened";
    });
    socket.addEventListener('message', (e) => {
        // if (e.data.slice(0,3) == "|||") {
        //     document.getElementById("status").innerHTML = e.data.slice(0,3);
        // } 
        // else {

            
            //last 2 digit
            // count = e.data.slice(-2);

            // image sliced
            // e.data = e.data.slice(0, -4);
            let ctx = msg.getContext("2d");
            let image = new Image();
            image.src = URL.createObjectURL(e.data);
            image.addEventListener("load", (e) => {
                ctx.drawImage(image, 0, 0, msg.width, msg.height);
            });
        // }
    });
}
