import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;
import java.util.zip.GZIPInputStream;
import java.lang.Thread;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService; 
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.*;


import transforms.*;

import com.google.gson.*;

import spoon.Launcher;
import spoon.reflect.cu.*;
import spoon.reflect.CtModel;
import spoon.reflect.cu.position.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import spoon.OutputType;
import spoon.processing.AbstractProcessor;
import spoon.support.JavaOutputProcessor;
import spoon.support.compiler.VirtualFile;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.DefaultJavaPrettyPrinter;
import spoon.reflect.visitor.PrettyPrinter;

class TransformFileTask implements Runnable {	
	static AtomicInteger counter = new AtomicInteger(0); // a global counter

	String split;
	ArrayList<VirtualFile> inputs;
	boolean doDepthK;
	int K;
	int NUM_SAMPLES; 
	int chunkID;


	TransformFileTask(String split, ArrayList<VirtualFile> inputs, boolean depthK, int k, int numSamples, int chunkID) {
		this.split = split;
		this.inputs = inputs;
		this.doDepthK = depthK;
		this.K = k;
		this.NUM_SAMPLES = numSamples;
		this.chunkID = chunkID;
	}

	private Launcher buildLauncher(AverlocTransformer transformer) {
		Launcher launcher = new Launcher();

		launcher.getEnvironment().setCopyResources(false);
		launcher.getEnvironment().setNoClasspath(true);
		launcher.getEnvironment().setShouldCompile(false);
		launcher.getEnvironment().setLevel("OFF");
		launcher.getEnvironment().setOutputType(OutputType.NO_OUTPUT);

		launcher.addProcessor(transformer);

		launcher.setSourceOutputDirectory(
			String.format("/mnt/raw-outputs/%s/%s", transformer.getOutName(), split)
		);

		return launcher;
	}

	public void outputFiles(CtModel model, AverlocTransformer transformer) {

		for (CtClass outputClass : model.getElements(new TypeFilter<>(CtClass.class))) {
			if (transformer.changes(outputClass.getSimpleName())) {
				Path path = Paths.get(
					String.format(
						"/mnt/raw-outputs/%s/%s/%s.java",
						transformer.getOutName(),
						split,
						outputClass.getSimpleName().replace("WRAPPER_", "")
					)
				);

				if (!path.getParent().toFile().exists()) {
					path.getParent().toFile().mkdirs();
				}

				try {
					File file = path.toFile();
					file.createNewFile();
		
					PrintStream stream = new PrintStream(file);
					stream.print(outputClass.toString());
				} catch (Exception ex) {
					System.out.println("Failed to save: " + path.toString());
					ex.printStackTrace(System.out);
					continue;
				}
			}
		}
		if (transformer.siteMap.size() != 0) {
			Path siteMapPath = Paths.get(
				String.format("/mnt/outputs/%s/%s_site_map/%s.json", transformer.getOutName(), split, this.chunkID)
				);
			if (!siteMapPath.getParent().toFile().exists()) {
				siteMapPath.getParent().toFile().mkdirs();
			}
			try {
				FileWriter outFile = new FileWriter(siteMapPath.toString());
                // Gson gson = new Gson();
                Gson gson = new GsonBuilder().disableHtmlEscaping().create();
				outFile.write(gson.toJson(transformer.siteMap));
				outFile.flush();
				outFile.close();
			} catch (Exception ex) {
				System.out.println("Failed to save: " + siteMapPath.toString());
			}
		}		

		
	}

	public void run() {
		ArrayList<AverlocTransformer> transformers = new ArrayList<AverlocTransformer>();

		transformers.add(new Identity());

	Random rand = new Random();

		if (doDepthK) {
			// Take NUM_SAMPLES many sequences of DEPTH length
			for (int s = 0; s < NUM_SAMPLES; s++) {
				ArrayList<AverlocTransformer> subset = new ArrayList<AverlocTransformer>();
				
				// Random, allow duplciates, do depth K
				for (int i = 0; i < K; i++) {
					int choice = rand.nextInt(8);

					if (choice == 0){
						subset.add(new AddDeadCode(i, false));
					} else if (choice == 1) {
						subset.add(new WrapTryCatch(i, false));
					} else if (choice == 2) {
						subset.add(new UnrollWhiles(i, false));
					} else if (choice == 3) {
						subset.add(new InsertPrintStatements(i, false));
					} else if (choice == 4) {
						subset.add(new RenameFields(i, false));
					} else if (choice == 5) {
						subset.add(new RenameLocalVariables(i, false));
					} else if (choice == 6) {
						subset.add(new RenameParameters(i, false));
					} else if (choice == 7) {
						subset.add(new ReplaceTrueFalse(i, false));
					}
				}

				transformers.add(new Sequenced(
					subset,
					"depth-" + Integer.toString(K) + "-sample-" + Integer.toString(s + 1), 1
				));

			}
		} else {
			// transformers.add(new AddDeadCode(1, true));
			// transformers.add(new WrapTryCatch(1, true));
			// transformers.add(new UnrollWhiles(1, true));
			// transformers.add(new InsertPrintStatements(1, true));
			// transformers.add(new RenameFields(1, true));
			// transformers.add(new RenameLocalVariables(1, true));
			// transformers.add(new RenameParameters(1, true));
			// transformers.add(new ReplaceTrueFalse(1, false));
			// transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new RenameParameters(1, true), new RenameLocalVariables(1, true))), "renamevar-param"));
			// transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new RenameParameters(1, false), new RenameLocalVariables(1, false))), "renamevar-param-single"));
			transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new RenameLocalVariables(1, true), new RenameParameters(1, true), new RenameFields(1, true), new ReplaceTrueFalse(1, true), new InsertPrintStatements(1, true), new AddDeadCode(1, true))), "transforms.Combined", 1));
			transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new RenameLocalVariables(1, true), new RenameParameters(1, true), new RenameFields(1, true), new ReplaceTrueFalse(1, true))), "transforms.Replace", 1));
			transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new InsertPrintStatements(1, true), new AddDeadCode(1, true))), "transforms.Insert", 1));

            // transformers.add(new Sequenced(new ArrayList<AverlocTransformer>(Arrays.asList(new RenameLocalVariables(1, true), new RenameParameters(1, true), new InsertPrintStatements(1, true), new ReplaceTrueFalse(1, true), new RenameFields(1, true), new AddDeadCode(1, true), new UnrollWhiles(1, true),  new WrapTryCatch(1, true))), "transforms.Combined", 1));
		}

		System.out.println(String.format("     + Have %s tranforms.", transformers.size()));

		ArrayList<String> failures = new ArrayList<String>();
		for (AverlocTransformer transformer : transformers) {
			try {
				Launcher launcher = buildLauncher(transformer);

				for (VirtualFile input : inputs) {
					launcher.addInputResource(input);
				}

				CtModel model = launcher.buildModel();
				model.processWith(transformer);

				outputFiles(model, transformer);
			} catch (Exception ex1) {
                System.out.println(ex1);
				for (VirtualFile singleInput : inputs) {
					if (failures.contains(singleInput.getName())) {
						continue;
					}

					try {
						Launcher launcher = buildLauncher(transformer);
						launcher.addInputResource(singleInput);

						CtModel model = launcher.buildModel();
						model.processWith(transformer);

						outputFiles(model, transformer);
					} catch (Exception ex2) {
                        System.out.println(ex2);
						System.out.println(
							String.format(
								"     * Failed to build model for: %s",
								singleInput.getName()
							)
						);
						failures.add(singleInput.getName());
					}
				}
			}
		}

		int finished = counter.incrementAndGet();
		System.out.println(String.format("     + Tasks finished: %s", finished));
	
	}
}

public class Transforms {
	private static Callable<Void> toCallable(final Runnable runnable) {
	return new Callable<Void>() {
		@Override
		public Void call() {
					try {
			runnable.run();
					} catch (Exception e) {
						e.printStackTrace(System.err);
						System.err.println(e.toString());
					}
						return null;
		}
	};
	} 
	
	private static <T> ArrayList<ArrayList<T>> chopped(ArrayList<T> list, final int L) {
	ArrayList<ArrayList<T>> parts = new ArrayList<ArrayList<T>>();
	final int N = list.size();
	for (int i = 0; i < N; i += L) {
		parts.add(new ArrayList<T>(
			list.subList(i, Math.min(N, i + L)))
		);
	}
	return parts;

  }

	private static ArrayList<Callable<Void>> makeTasks(String split, String maybeDepth, String numSamples) {
		try {
			// Return list of tasks
			ArrayList<Callable<Void>> tasks = new ArrayList<Callable<Void>>();
		
			// The file this thread will read from
			InputStream fileStream = new FileInputStream(String.format(
				"/mnt/inputs/%s.jsonl.gz",
				split
			));

			// File (gzipped) -> Decoded Stream -> Lines
			InputStream gzipStream = new GZIPInputStream(fileStream);
			Reader decoder = new InputStreamReader(gzipStream, "UTF-8");
			BufferedReader buffered = new BufferedReader(decoder);

			// From gzip, create virtual files
			String line;
			JsonParser parser = new JsonParser();
			ArrayList<VirtualFile> inputs = new ArrayList<VirtualFile>();
			while ((line = buffered.readLine()) != null) {
				JsonObject asJson = parser.parse(line).getAsJsonObject();

				inputs.add(new VirtualFile(
					asJson.get("source_code").getAsString().replace(
						"class WRAPPER {",
						String.format(
							"class WRAPPER_%s {",
							asJson.get("sha256_hash").getAsString()
						)
					),
					String.format("%s.java", asJson.get("sha256_hash").getAsString())
				));
			}

			boolean doDepthK = maybeDepth != null && maybeDepth != "" && maybeDepth.matches("\\d+");

			int chunkID = 0;

			for (ArrayList<VirtualFile> chunk : chopped(inputs, 3000)) {
				tasks.add(toCallable(new TransformFileTask(
					split,
					chunk,
					doDepthK,
					doDepthK ? Integer.parseInt(maybeDepth) : 1,
					doDepthK ? Integer.parseInt(numSamples) : 1,
					chunkID
				)));
				chunkID += 1;
			}

			return tasks;
		}
		catch (Exception ex) {
			ex.printStackTrace();
			System.out.println(ex.toString());
			return new ArrayList<Callable<Void>>();
		}
	}

	public static void joinChunks(String pathToMaps) {
		File fMaps = new File(pathToMaps);
		String[] paths = fMaps.list();
		// System.out.println(paths);
		for (String path : paths) {
			if (path.contains("site_map") && !path.contains("site_map.json")){
				File f = new File(pathToMaps+path);
				String[] pathnames = f.list();
				HashMap allSites = new HashMap<String, HashMap>();
				Gson gson = new GsonBuilder().disableHtmlEscaping().create();;
				for (String pathname : pathnames) {
					try {
						Reader reader = Files.newBufferedReader(Paths.get(pathToMaps+path+"/"+pathname));
						HashMap sites = gson.fromJson(reader, HashMap.class);
						allSites.putAll(sites);
						File fname = new File(pathToMaps+"/"+path+"/"+pathname);
						fname.delete();
						reader.close();
					} catch (Exception ex) {
						ex.printStackTrace();
					}
				}
				Path siteMapPath = Paths.get(pathToMaps+path+".json");
				if (!siteMapPath.getParent().toFile().exists()) {
					siteMapPath.getParent().toFile().mkdirs();
				}
				try {
					FileWriter outFile = new FileWriter(siteMapPath.toString());
					// Gson gson = new Gson();
					outFile.write(gson.toJson(allSites));
					outFile.flush();
					outFile.close();
				} catch (Exception ex) {
					System.out.println("Failed to save: " + siteMapPath.toString());
				}
			File fname = new File(pathToMaps+"/"+path);
			fname.delete();
			}
		}
	
	
	}

	public static void main(String[] args) {
		try {
			ArrayList<Callable<Void>> allTasks = new ArrayList<Callable<Void>>();

			if (System.getenv("AVERLOC_JUST_TEST").equalsIgnoreCase("true")) {
				System.out.println("Populating tasks...");
				System.out.println("   - Adding from test split...");
				allTasks.addAll(Transforms.makeTasks("test", System.getenv("DEPTH"), System.getenv("NUM_SAMPLES")));
				System.out.println(String.format("     + Now have %s tasks...", allTasks.size()));
			} else {
				System.out.println("Populating tasks...");
				System.out.println("   - Adding from test split...");
				allTasks.addAll(Transforms.makeTasks("test", System.getenv("DEPTH"), System.getenv("NUM_SAMPLES")));
				System.out.println(String.format("     + Now have %s tasks...", allTasks.size()));
				System.out.println("   - Adding from train split...");
				allTasks.addAll(Transforms.makeTasks("train", System.getenv("DEPTH"), System.getenv("NUM_SAMPLES")));
				System.out.println(String.format("     + Now have %s tasks...", allTasks.size()));
				System.out.println("   - Adding from valid split...");
				allTasks.addAll(Transforms.makeTasks("valid", System.getenv("DEPTH"), System.getenv("NUM_SAMPLES")));
				System.out.println(String.format("     + Now have %s tasks...", allTasks.size()));
			}

			System.out.println("   - Running in parallel with 64 threads...");
			ExecutorService pool = Executors.newFixedThreadPool(64);

			// allTasks.get(4).call();
			pool.invokeAll(allTasks);
			// pool.invokeAll(allTasks.stream().limit(10).collect(Collectors.toList()));

			pool.shutdown(); 
			String[] tNames = {"transforms.Combined", "transforms.Insert", "transforms.Replace", 
								"transforms.RenameLocalVariables", "transforms.RenameParameters", "renamevar-param", 
								"renamevar-param-single", "transforms.AddDeadCode", "transforms.RenameFields",
								"transforms.InsertPrintStatements", "transforms.Identity", "transforms.UnrollWhiles", "transforms.ReplaceTrueFalse", "transforms.WrapTryCatch"};
			for (String transform : tNames) {
				String pathStr = String.format("/mnt/outputs/%s/", transform);
				Path path = Paths.get(pathStr);
				if (path.toFile().exists()) {
					joinChunks(pathStr);
				}
				
			}
			System.out.println("   + Done!");
		} catch (Exception ex) {
			ex.printStackTrace();
			System.out.println(ex.toString());
		}
	}


}