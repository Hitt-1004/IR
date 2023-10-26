import java.io.IOException;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.ByteBuffersDirectory;

public class Main {
    public static void main(String[] args) throws IOException, ParseException {
        // 0. Specify the analyzer for tokenizing text.
        //    The same analyzer should be used for indexing and searching
        StandardAnalyzer analyzer = new StandardAnalyzer();

        // 1. create the index
        Directory index = new ByteBuffersDirectory();

        IndexWriterConfig config = new IndexWriterConfig(analyzer);

        IndexWriter w = new IndexWriter(index, config);
        addDoc(w, "1) Make a small network with a few PCs, one web server, a switch, and a router (just to give an idea of how to assign a default gateway). Make sure that all the devices are connected (use straight-through cables as we are connecting different devices). Create a web page that should contain your name and your roll number.", "193398817");
        addDoc(w, "Make a complex network having at least two different network addresses. You can use many PCs, switches, and routers. Again create a web page through a Server. Note that this time the PC used to display the web page should be connected to another network address than a web server.", "55320055Z");
        addDoc(w, "Dear SIH SPOCs,Warm greetings from the Ministry of Education’s Innovation Cell (MIC) and the All India Council for Technical Education (AICTE).We want to extend our heartfelt gratitude for your enthusiasm and commitment to the Smart India Hackathon 2023 (SIH 2023). Your dedication to innovation and problem-solving is truly inspiring.We are writing to inform you of an important update regarding the SIH 2023 Problem Statements (PSs). Unfortunately, due to some unforeseen reasons by the Ministry, they have regrettably withdrawn their support for the Ministry of Micro, Small & Medium Enterprises (MSME) and Government of Jammu & Kashmir challenge categories. While this news is disappointing, we remain fully committed to ensuring that SIH 2023 continues to be a platform for your innovative ideas and solutions.", "55320055Z");
        addDoc(w, "Respected sir, I request to change timing of Saturday OPD to 12:15 - 1:15 pm , instead of 11:30 pm. rest timings would be same for weekdays.Thanks and regards.", "55320055Z");
        addDoc(w, "dear sir, PFA is a D.O. letter with a request to promote the portal and disseminate the special course module among the students and teachers of your University/Institution. ", "55320055Z");
        addDoc(w, "Dear Students,Transport schedule for next week, 16.10.23 to 31.10.2023 is attached with this mail.Lunch Venue is also mentioned. @ Bhojraj Joshi.Lunch would be available between 12:00 to 14:00. All buses would have Batch No written .Board the buses accordingly.All buses would leave Hostel as per Schedule.In case of any query do contact Sh. Gohil Bhupendrasinh, Rahul Nawanath (2022) and Sanidhya Kumar (2021).\n", "55320055Z");
//        addDoc(w, "Make a complex network having at least two different network addresses. You can use many PCs, switches, and routers. Again create a web page through a Server. Note that this time the PC used to display the web page should be connected to another network address than a web server.", "55320055Z");
        //        addDoc(w, "Managing Gigabytes", "55063554A");
//        addDoc(w, "The Art of Computer Science", "9900333X");
        w.close();

        // 2. query
        String querystr = args.length > 0 ? args[0] : "the";

        // the "title" arg specifies the default field to use
        // when no field is explicitly specified in the query.
        Query q = new QueryParser("title", analyzer).parse(querystr);

        // 3. search
        int hitsPerPage = 10;
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        displaySearchResults(searcher, q, hitsPerPage);

        TopDocs docs = searcher.search(q, hitsPerPage);
        ScoreDoc[] hits = docs.scoreDocs;

        // 4. display results
        System.out.println("Found " + hits.length + " hits.");
        for(int i=0;i<hits.length;++i) {
            int docId = hits[i].doc;
            Document d = searcher.doc(docId);
            System.out.println((i + 1) + ". " + d.get("isbn") + "\t" + d.get("title"));
        }

        // reader can only be closed when there
        // is no need to access the documents any more.
        reader.close();
    }

    private static void addDoc(IndexWriter w, String title, String isbn) throws IOException {
        Document doc = new Document();
        doc.add(new TextField("title", title, Field.Store.YES));

        // use a string field for isbn because we don't want it tokenized
        doc.add(new StringField("isbn", isbn, Field.Store.YES));
        w.addDocument(doc);
    }

    public static void displaySearchResults(IndexSearcher searcher, Query query, int hitsPerPage) throws IOException {
        TopDocs docs = searcher.search(query, hitsPerPage);
        ScoreDoc[] hits = docs.scoreDocs;

        System.out.println("Found " + hits.length + " hits.");

        for (int i = 0; i < hits.length; ++i) {
            int docId = hits[i].doc;
            Document d = searcher.doc(docId);

            // Get the relevance score for the current document
            float score = hits[i].score;

            System.out.println((i + 1) + ". Rank: " + (i + 1) + "\tRelevance Score: " + score);
            System.out.println("   ISBN: " + d.get("isbn") + "\tTitle: " + d.get("title"));
        }
    }
}
