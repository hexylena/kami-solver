var w = 500,
    h = 400;

function loadJSON(callback) {
    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', 'force.json', true);
    xobj.onreadystatechange = function () {
        if (xobj.readyState == 4 && xobj.status == "200") {
            callback(xobj.responseText);
        }
    }
    xobj.send(null);
};

loadJSON(function(response) {
    var data = JSON.parse(response);
    for(var i = 0; i < data.length; i++){
        var node = $("#main").append('<div id="n' + i + '"></div>');
        var d = $("#n" + i);
        d.append('<h1>Step ' + i +'</h1>');

        d.append('<div id="r' + i + '" class="row"></div>');
        var r = $("#r" + i);
        r.append('<div id="l' + i + '" class="col-md-6"><div id="c' + i + '"></div></div>');

        var table = layoutTable(data[i]);

        var n = $("#c" + i);
        var x = layoutData(n[0], data[i]);

        r.append('<div class="col-md-6">' +  table + '</div>');
    }
});

function layoutTable(data){
    table = {};

    for(var i=0; i < data.nodes.length; i++){
        // group
        var rgb = data.nodes[i].colour;
        for(var j=0; j<data.nodes[i].points.length; j++){
            var point = data.nodes[i].points[j],
                a = point[0],
                b = point[1];

            if(!table[a]){ table[a] = {}; }
            table[a][b] = rgb
        }
    }
    tableString = "<table>"

    Object.keys(table).forEach(function(i) {
        var row = "<tr>";
        Object.keys(table[i]).forEach(function(j) {
            var colour = "rgb(" + table[i][j][0] + "," + table[i][j][1] + "," + table[i][j][2] + ")"
            row = row + '<td><span style="background:' + colour + ';padding:10px;"></span></td>'
        });
        tableString = tableString + row + "</tr>\n";
    });

    return tableString + "</table>"
}

function layoutData(node, json){
  var vis = d3.select(node)
    .append("svg:svg")
      .attr("width", w)
      .attr("height", h);

  var force = d3.layout.force()
      .charge(-80)
      .linkDistance(30)
      .nodes(json.nodes)
      .links(json.links)
      .size([w, h])
      .start();

  var link = vis.selectAll("line.link")
      .data(json.links)
      .enter().append("svg:line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return Math.sqrt(d.value); })
      .attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  var node = vis.selectAll("circle.node")
      .data(json.nodes)
      .enter().append("svg:circle")
      .attr("class", "node")
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; })
      .attr("r", function(d){ return Math.log10(d.size + 1) * 10; })
      .style("fill", function(d) {
          if(!d.colour){
               console.log(d);
               return 'red';
          }
          colour = 'rgb(' + d.colour[0] + ',' + d.colour[1] + ','+ d.colour[2] + ')';
          return colour;
      })
      .text(function(d){ return d.id; })
      .call(force.drag);

  vis.style("opacity", 1e-6)
    .transition()
      .duration(1000)
      .style("opacity", 1);

  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  });
};
