 $(document).ready(function(){

    onstart_page();
    function onstart_page(){

        $.ajax({
            type:'POST',
            url:'/detectStatus',
            cache: false,
            timeout: 10000,
            error: function(){
             console.log("can't check detect mode");
         },
         success: function (data) {
            console.log(data);
            if(data == 'on'){
                $('#d-content').html(' <button id="detect-mode" class="btn btn-danger" style="background-color: #28a745;" onclick="toggleDT()">is On</button>');
                $('div[name="live-content"').html('<button class="btn btn-info bne1 formButtonfront" name="live" type="submit" disabled="true">Live</button>');
                console.log('status '+data);
            }
            else if(data == 'off'){
                $('#d-content').html('<button id="detect-mode" class="btn btn-success" style="background-color: #dc3545;" onclick="toggleDT()">is OFF</button>');
                $('div[name="live-content"').html('<button class="btn btn-info bne1 formButtonfront" name="live" type="submit">Live</button>');
            }
            $('button[name="live"]').click(function(){
                $.ajax({
                   type:'GET',
                   url:'/live_camera',
                   cache: false,
                   timeout: 10000,
                   success: function (data) {
                    console.log('pop');
                    $("body").html(data);
                }
            })
            })
        }
    });
    }

}) 


 function postToggleDetect(mode){

    $.ajax({
       type:'POST',
       url:'/mode/'+mode,
       cache: false,
       timeout: 10000,
       error: function(){
           console.log("can't toggle detect mode");
       },
       success: function (data) {
        console.log(data);
    }
})
}

function toggleDT(){
    var live_btn = $('button[name="live"');
    $.ajax({
       type:'POST',
       url:'/detectStatus',
       cache: false,
       timeout: 10000,
       async: false,
       error: function(){
           console.log("can't check detect mode");
       },
       success: function (data) {
        if(data == 'off'){
            postToggleDetect('on')
            $('#d-content').html(' <button id="detect-mode" class="btn btn-danger" style="background-color: #28a745;" onclick="toggleDT()">is On</button>');
            live_btn.attr("disabled", true);
        }
        else if(data == 'on'){
            postToggleDetect('off')
            $('#d-content').html('<button id="detect-mode" class="btn btn-success" style="background-color: #dc3545;" onclick="toggleDT()">is OFF</button>');
            live_btn.attr("disabled", false);
        }
    }
})
}


$('button[id="set_confidence_btn"]').click(function(){
    var tmp = $("#_confidence").val();
    $.ajax({type:'POST',
        url:'/set/confidence',
        cache: false,
        data: {'confidence':tmp},
        timeout: 10000,
        error: function(){
            console.log('OK');
            return true;
        },
        success: function(msg){
            console.log("set new confidence ->"+ tmp);
        }});
})

$('button[id="set_frame_btn"]').click(function(){
    var tmp = $("#_frame").val();
    $.ajax({type:'POST',
        url:'/set/frame',
        cache: false,
        data: {'frame':tmp},
        timeout: 10000,
        error: function(){
            console.log('OK');
            return true;
        },
        success: function(msg){
            console.log("set new frame ->" + tmp);     
        }});
})

