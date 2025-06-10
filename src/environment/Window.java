package environment;

import java.nio.FloatBuffer;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GLCapabilities;
import org.lwjgl.system.MemoryUtil;

public class Window {
	int width;
	int height;

	long window;

	int maxWidth, maxHeight;

	GLCapabilities capabilities;

	int program;
	
	//uniform locations
	int color;
	int displacement;

	public Window(int maxWidth, int maxHeight) {
		this.maxWidth = maxWidth;
		this.maxHeight = maxHeight;
	}

	public void init() {
		// do opengl/glfw init stuff
		boolean res = GLFW.glfwInit();
		// fail information
		if (res == false) {
			System.err.println("GLFW couldn't initialize. Please try again");
			System.exit(0);
		}

		GLFW.glfwDefaultWindowHints();

		GLFW.glfwWindowHint(GLFW.GLFW_VISIBLE, GL11.GL_FALSE);
		GLFW.glfwWindowHint(GLFW.GLFW_RESIZABLE, GL11.GL_FALSE);
		GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 4);
		GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 3);
		GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

		window = GLFW.glfwCreateWindow(800, 800, "Pong", MemoryUtil.NULL, MemoryUtil.NULL);

		if (window == MemoryUtil.NULL) {
			System.err.println("Something went wrong with Window Creation. Please try again");
			System.exit(0);
		}

		GLFW.glfwSetWindowPos(window, 100, 100);
		GLFW.glfwSetCursorPos(window, 400, 480);

		width = maxWidth / 2;
		height = maxHeight / 2;

		double[] x = new double[2];
		double[] y = new double[2];
		GLFW.glfwGetCursorPos(window, x, y);

		GLFW.glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
			// check if the key was escape and the action was release
			if (key == GLFW.GLFW_KEY_ESCAPE && action == GLFW.GLFW_RELEASE) {
				// then we should set that the window should close to true
				GLFW.glfwSetWindowShouldClose(window, true);
			}
		});

		GLFW.glfwMakeContextCurrent(window);

		capabilities = GL.createCapabilities();

		GL11.glClearColor(0, 0, 0, 0);

		GLFW.glfwWindowHint(GLFW.GLFW_VISIBLE, GLFW.GLFW_TRUE);
		GLFW.glfwShowWindow(window);

		GLFW.glfwSwapInterval(1);
		// init a basic image shader that draws an image
		int sid = GL20.glCreateShader(GL20.GL_VERTEX_SHADER);
		GL20.glShaderSource(sid, ""
				+ "#version 460 core\n"
				+ "layout(location = 0) in vec2 pos;\n\n"
				+ "out vec2 Pos;\n\n"
				+ "uniform vec2 displacement;\n\n"
				+ "void main() {\n"
				+ "  gl_Position = vec4(pos+displacement, 1.0, 1.0);\n"
				+ "  Pos = pos;\n"
				+ "}"
				);

		GL20.glCompileShader(sid);
		if (GL20.glGetShaderi(sid, GL20.GL_COMPILE_STATUS) == GL20.GL_FALSE) {
			System.err.println("vertex shader compile fail");
			System.err.println(GL20.glGetShaderInfoLog(sid, 1000));
			GL20.glDeleteShader(sid);
			System.exit(1);
		}

		int fid = GL20.glCreateShader(GL20.GL_FRAGMENT_SHADER);
		GL20.glShaderSource(fid, ""
				+ "#version 460 core\n"
				+ "in vec2 Pos;\n\n"
				+ "uniform vec3 color;\n\n"
				+ "void main() {\n"
				+ "  gl_FragColor = vec4(color, 1.0);\n"
				+ "}"
				);
		GL20.glCompileShader(fid);
		if (GL20.glGetShaderi(fid, GL20.GL_COMPILE_STATUS) == GL20.GL_FALSE) {
			System.err.println("fragment shader compile fail");
			System.err.println(GL20.glGetShaderInfoLog(fid, 1000));
			GL20.glDeleteShader(fid);
			System.exit(1);
		}

		program = GL20.glCreateProgram();
		GL20.glAttachShader(program, sid);
		GL20.glAttachShader(program, fid);

		GL20.glLinkProgram(program);
		if (GL20.glGetProgrami(program, GL20.GL_LINK_STATUS) == GL20.GL_FALSE) {
			System.err.println("shader link program fail");
			System.err.println(GL20.glGetProgramInfoLog(program, 1000));
			GL20.glDeleteProgram(program);
			GL20.glDeleteShader(sid);
			GL20.glDeleteShader(fid);
			System.exit(1);
		}
		GL20.glDeleteShader(fid);
		GL20.glDeleteShader(sid);
		GL20.glValidateProgram(program);

		GL30.glUseProgram(program);
		
		// add color uniform
		color = GL30.glGetUniformLocation(program, "color");
		if (color < 0) {
			// image doesn't exist
			System.err.println("fragment doesn't have color (fail)");
		}
		
		// add color uniform
		displacement = GL30.glGetUniformLocation(program, "displacement");
		if (displacement < 0) {
			// image doesn't exist
			System.err.println("vertex doesn't have displacement (fail)");
		}
	}

	public void update() {
		GLFW.glfwSwapBuffers(window);
		GLFW.glfwPollEvents();
	}

	public void cleanup() {
		GLFW.glfwDestroyWindow(window);
		GLFW.glfwTerminate();
	}

	public boolean shouldClose() {
		return GLFW.glfwWindowShouldClose(window);
	}
	
	public void clear() {
		GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT);
	}
	
	public void draw(Drawable d) {
		GL20.glUseProgram(program);
		GL30.glBindVertexArray(d.vao);
		GL20.glEnableVertexAttribArray(0);

		GL20.glUniform2f(displacement, d.xDisplacement, d.yDisplacement);
		GL20.glUniform3f(color, d.red, d.green, d.blue);

		GL20.glDrawArrays(GL11.GL_TRIANGLES, 0, d.vertexCount);

		GL30.glDisableVertexAttribArray(0);
		GL30.glBindVertexArray(0);
	}
	
	public boolean getKey(int key) {
		return GLFW.glfwGetKey(window, key) == GLFW.GLFW_PRESS;
	}
}

class Drawable {
	float[] shape;
	float xDisplacement;
	float yDisplacement;
	float red;
	float green;
	float blue;
	
	int vao;
	int vbo;
	
	int vertexCount;
	
	public Drawable() {
		//init white, 0 displacement, and rectangular shape
		xDisplacement = yDisplacement = 0.0f;
		
		red = green = blue = 1.0f;
		
		float[] vertices = { 
				-1.0f, 1.0f, 
				-1.0f, -1.0f,
				1.0f, 1.0f, 

				1.0f, 1.0f, 
				-1.0f, -1.0f,
				1.0f, -1.0f,};
		
		shape = vertices;
	}
	
	public void load() {
		FloatBuffer buff = MemoryUtil.memAllocFloat(shape.length);
		buff.put(shape).flip();

		vao = GL30.glGenVertexArrays();
		GL30.glBindVertexArray(vao);

		vbo = GL15.glGenBuffers();
		GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo);
		GL15.glBufferData(GL15.GL_ARRAY_BUFFER, buff, GL15.GL_STATIC_DRAW);
		GL20.glVertexAttribPointer(0, 2, GL11.GL_FLOAT, false, Float.BYTES * (2), 0);

		GL30.glBindVertexArray(0);
		GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0);
		vertexCount = 6;
		MemoryUtil.memFree(buff);
	}
	
	public void reload() {
		//first delete
		cleanup();
		
		//then load
		load();
	}
	
	public void cleanup() {
		GL30.glDisableVertexAttribArray(vao);
		GL20.glBindBuffer(GL20.GL_ARRAY_BUFFER, 0);
		GL20.glDeleteBuffers(vbo);
		GL30.glBindVertexArray(0);
	}
	
	
}
