package environment;
public class Main {
	public static void main(String[] args) {
		Window window = new Window(1920, 1080);

		window.init();

		Pong pong = new Pong();
		pong.w = window;
		pong.play();
	}
}
