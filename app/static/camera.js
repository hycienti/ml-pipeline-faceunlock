async function startCamera(video, errorEl) {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    video.hidden = true;
    errorEl.hidden = false;
  }
}

function snap(video) {
  return new Promise((resolve, reject) => {
    if (!video.videoWidth) return reject(new Error("no camera frame"));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
  });
}
