document.addEventListener('DOMContentLoaded', function () {
    var item = localStorage.getItem('temp_model_ID');
    var select = document.getElementById("temp_model_ID");
    select.value = item;
});

function submitForm() {
    var select = document.getElementById("temp_model_ID");
    var value = select.options[select.selectedIndex].value;
    localStorage.setItem('temp_model_ID', value);
}