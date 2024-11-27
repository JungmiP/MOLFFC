package spring.lstm.controller;

import jakarta.servlet.http.*;

import java.nio.charset.Charset;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.json.JSONObject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MyController {

	@ResponseBody
	@RequestMapping(value="/sendImg", method=RequestMethod.POST)
	public String webcam(@RequestBody String data) throws Exception {

		System.out.println("data="+data);
		
		JSONObject dataObject = new JSONObject(data);
		System.out.println("dataObject="+dataObject);
		
		String cam_data = ((String) dataObject.getString("img_data"));
		System.out.println("cam_data="+cam_data);
		
		JSONObject restSendData = new JSONObject();
		restSendData.put("data", cam_data);
		System.out.println("restSendData="+restSendData);
		
		HttpPost httpPost = new HttpPost("http://localhost:5000/calScore");
		httpPost.addHeader("Content-Type", "application/json;charset=utf-8");
		
		StringEntity stringEntity = new StringEntity(restSendData.toString());
		
		httpPost.setEntity(stringEntity);
		
		CloseableHttpClient httpclient = HttpClients.createDefault();
		CloseableHttpResponse response2 = httpclient.execute(httpPost);
		
		String yolo_message = EntityUtils.toString(response2.getEntity(), Charset.forName("UTF-8"));
		
		return yolo_message;
	}
	
}
