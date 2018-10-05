$(document).ready(function() {

    // javascript code to enable download links of notebooks and scripts *.py
    // https://stackoverflow.com/questions/2304941
    if (document.readyState == "interactive") {
         document.getElementsByClassName("last")[0].children[0].children[0].setAttribute("target", "_blank")
         document.getElementsByClassName("last")[0].children[2].children[1].setAttribute("download", "")
         document.getElementsByClassName("last")[0].children[2].children[2].setAttribute("download", "")
     }

    // add info note for past releases
    var verFile = new XMLHttpRequest();
    verFile.open("GET", "http://docs.gammapy.org/stable/index.html", true);
    verFile.onreadystatechange = function() {
      if (verFile.readyState === 4) {  // makes sure the document is ready to parse.
        if (verFile.status === 200) {  // makes sure it's found the file.
          var allText = verFile.responseText;
          var match =  allText.match(/url=\.\.\/(.*)"/i);
          var version = match[1];
          var note = '<div class="admonition note"><p class="first admonition-title">Note</p>'
          note += '<p class="last">You are not reading the most up to date version of Gammapy '
          note += 'documentation.<br/>Access the <a href="http://docs.gammapy.org/'
          note += version
          note += '/">latest stable version v'
          note += version
          note += '</a> or the <a href="http://gammapy.org/news.html#releases">list of Gammapy releases</a>.</p></div>'

          var url = window.location.href
          if (url.includes("/dev/") == false && url.includes(version) == false ) {
              var divbody = document.querySelectorAll('[role="main"]');
              var divnote = document.createElement("div");
              divnote.innerHTML = note;
              divbody[0].insertBefore(divnote, divbody[0].firstChild);
          }
        }
      }
    }
    verFile.send(null);

});
