def project = getProject()
//def outputDir = buildFilePath(PROJECT_BASE_DIR, 'json')
def outputDir = buildFilePath("/Users/oliverester/Projekte/PräDikNika/qupath_export/CD8_CD20")
mkdirs(outputDir)
boolean prettyPrint = false // set to false for smaller files, to true for human-readable jsons
def gson = GsonTools.getInstance(prettyPrint)

for (entry in project.getImageList()) {
    def name = GeneralTools.getNameWithoutExtension(entry.getImageName())
    def path = buildFilePath(outputDir, name + ".json")
    def annotations = entry.readHierarchy().getAnnotationObjects()
    print entry.getImageName() + ', n. annotations: ' + annotations.size()
    def file = new File(path)
    file.write(gson.toJson(annotations))
}

