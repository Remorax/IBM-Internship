package fr.irit.generator;

public class Date {
	private int day;
	private int month;
	private int year;
	private int time;
	
	public Date() {
		year =(int)(Math.random()*28.0)+1990;
		month =(int)(Math.random()*11.0)+1;
		day =(int)(Math.random()*30.0)+1;
		time =(int)(Math.random()*24.0);
	}
	public Date(int time, int day, int month, int year) {
		this.time = time;
		this.day = day;
		this.month = month;
		this.year = year;
	}
	
	public String toString() {
		String strmonth =""+this.month;
		String strday =""+this.day;
		String strtime =""+this.time;
		if (this.month < 10) {
			strmonth="0"+this.month;
		}
		if (this.day < 10) {
			strday="0"+this.day;
		}
		if (this.time < 10) {
			strtime="0"+this.time;
		}
		
		
		return this.year+"-"+strmonth+"-"+strday+"T"+strtime+":00:00";
	}

	public Date takeOffDays( int days) {
		int newDay = this.day;
		int newMonth = this.month;
		int newYear = this.year;
		while(days > 365) {
			days-=365;
			newYear-=1;
		}
		while(days > 30) {
			days-=30;
			if(newMonth==1) {
				newMonth=12;
				newYear-=1;
			}
			else {
				newMonth-=1;
			}
			
		}
		if (newDay - days <= 0) {
			if(newMonth==1) {
				newMonth=12;
				newYear-=1;
			}
			else {
				newMonth-=1;
			}
			newDay=30+ newDay - days;
		}
		else {
			newDay-=days;
		}
		
		if(newMonth == 2 && newDay>28) {
			newDay = 28;
		}
		return new Date(this.time,newDay,newMonth,newYear);

	}
	
	public void setTime(int time) {
		this.time = time;
	}
}
