const msgerForm = get(".msger-inputarea");
const msgerInput = get(".msger-input");
const msgerChat = get(".msger-chat");
const debug = get(".debug");

// const BOT_MSGS = [
//   "Hi, how are you?",
//   "Ohh... I can't understand what you trying to say. Sorry!",
//   "I like to play games... But I don't know how to play!",
//   "Sorry if my answers are not relevant. :))",
//   "I feel sleepy! :("
// ];

// Icons made by Freepik from www.flaticon.com
const BOT_IMG = "https://image.flaticon.com/icons/svg/327/327779.svg";
const PERSON_IMG = "https://image.flaticon.com/icons/svg/145/145867.svg";
const BOT_NAME = "BOT";
const PERSON_NAME = "Guest";

msgerForm.addEventListener("submit", event => {
  event.preventDefault();

  const msgText = msgerInput.value;
  if (!msgText) return;

  appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
  msgerInput.value = "";

  botResponse(msgText);
});

function appendMessage(name, img, side, text) {
  //   Simple solution for small apps
  const msgHTML = `
    <div class="msg ${side}-msg">
      <div class="msg-img" style="background-image: url(${img})"></div>
      <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">${name}</div>
          <div class="msg-info-time">${formatDate(new Date())}</div>
        </div>
        <div class="msg-text">${text}</div>
      </div>
    </div>
  `;

  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}

function appendDebug(msg ="", intent_17 ="",intent_45 = "", ner="", model="", response="") {
  //   Simple solution for small apps
  const msgHTML = `
    <div class="card bg-light mb-3" style="max-width: 100%;">
        <div class="card-header">Thông tin</div>
        <div class="card-body">
          <p class="card-text">
              <strong>Message: </strong>${msg}
          </p>
          <p class="card-text">
              <strong>Model: </strong>${model}
          </p>
          <p class="card-text">
              <strong>Category: </strong>${intent_17}
          </p>
          <p class="card-text">
              <strong>Intent: </strong>${intent_45}
          </p>
          <p class="card-text">
              <strong>NER: </strong>${ner}
          </p>
          <p class="card-text">
              <strong>Response: </strong>${response}
          </p>
        </div>
    </div>
  `;

  debug.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}

function botResponse(text) {

  // const r = random(0, BOT_MSGS.length - 1);
  var req = new XMLHttpRequest();
  req.open("POST", "/process", false);
  var rawText = text;
  var dict = {text: rawText};
  console.log(dict);
  req.setRequestHeader("Content-Type", "application/json; charset=UTF-8");
  req.send(JSON.stringify(dict));
  var resp = JSON.parse(req.response).text_tagged;
  console.log(text);


  var model = document.getElementById("select_").value;

  const intent_17 = resp.results_17;
  const intent_45 = resp.results_45;
  appendDebug(text, intent_17,intent_45, resp.results_NER, model, resp.answer);

  const msgText = resp.answer !== "" ? resp.answer : "Đây là message của bot.";
  console.log(msgText);
  appendMessage(BOT_NAME, BOT_IMG, "left", msgText);
}

// Utils
function get(selector, root = document) {
  return root.querySelector(selector);
}

function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

function random(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}